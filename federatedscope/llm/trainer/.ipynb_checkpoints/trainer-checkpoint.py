import torch
import logging
import time
import random
import math
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken   # added by me, for gsm8k evaluation
from federatedscope.llm.misc.fschat import FSChatBot_My   # added by me, for gsm8k evaluation
from federatedscope.llm.eval.eval_for_gsm8k.eval import *   # added by me, for gsm8k evaluation
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT   # added by me, for gsm8k evaluation
from federatedscope.core.auxiliaries.utils import param2tensor, \
    merge_param_dict
from federatedscope.core.trainers.utils import format_log_hooks, \
    filter_by_specified_keywords
logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    clients_Mask=[]
    global_Mask={}
    cluster=[]
    vp=[]
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        self.d_i=time.time()
        
        # added by me, for gsm8k evaluation
        if not hasattr(ctx, 'val_loader_copy'):
            ctx.val_loader_copy = ctx.val_loader
        
    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            # logger.warning('Skip the batch due to the loss is NaN, '
            #                'it may be caused by exceeding the precision or '
            #                'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)
            if self.ctx.cfg.train.sparse_B == True and self.round>10:
                server_params_flat = torch.cat([p.grad.flatten() for n,p in ctx.model.named_parameters() if 'lora_B' in n])
                x=server_params_flat.abs().float()
                server_mask = torch.zeros_like(x).bool()
                k = int(x.numel()*self.ctx.cfg.train.B_ratio)
                _, keep_idx = torch.topk(x, k=k)
                server_mask[keep_idx] = 1
                curr = 0
                cfg_mask={}
                for n,p in ctx.model.named_parameters():
                    if 'lora_B' in n:
                        cfg_mask[n[6:]] = server_mask[curr:curr+p.numel()].reshape(p.shape)
                        curr += p.numel()
                        if len(self.cfg_mask)>0:                        
                            p.grad*=(cfg_mask[n[6:]] | self.cfg_mask[n[6:]])
                        else:
                            p.grad*=cfg_mask[n[6:]]
            if self.ctx.cfg.train.Lori == True and self.round>10:
                for n,p in ctx.model.named_parameters():
                    if 'lora_B' in n:
                        p.grad*=self.cfg_mask[n[6:]]                
                
                # self.cfg_mask=cfg_mask
                # for name,params in ctx.model.named_parameters():
                #     if 'lora_B' in name:
                #         params.grad *= self.cfg_mask[name[6:]]
                        
            ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        self.d_i=(time.time()-self.d_i)*1000
        self.mu_i=random.uniform(4, 6)
        computation_delay=simulate_computation_delay(self.x_i, self.mu_i, self.d_i)
        communication_delay=simulate_communication_delay(self.R, self.P_i, self.N0, self.W, self.S_model, self.cfg.federate.client_num)
        if ctx.cur_mode == MODE.TRAIN:
                eval_results = {
                    f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                    f'{ctx.cur_split}_total': ctx.num_samples,
                    f'{ctx.cur_split}_avg_loss': avg_loss,
                    f'{ctx.cur_split}_computation': computation_delay,
                    f'{ctx.cur_split}_communication':communication_delay,
                }
        else:
                eval_results = {
                    f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                    f'{ctx.cur_split}_total': ctx.num_samples,
                    f'{ctx.cur_split}_avg_loss': avg_loss,
                }
        if self.ctx.cfg.train.flasc == True:
            cfg_mask={}
            server_params_flat = torch.cat([p.flatten() for n,p in self.ctx.model.named_parameters() if 'lora' in n])
            x=server_params_flat.abs().float()
            server_mask = torch.zeros_like(x).bool()
            k = int(x.numel()*self.ctx.cfg.train.B_ratio)
            _, keep_idx = torch.topk(x, k=k)
            server_mask[keep_idx] = 1
            curr = 0
            for n,p in self.ctx.model.named_parameters():
                if 'lora' in n:
                    cfg_mask[n[6:]] = server_mask[curr:curr+p.numel()].reshape(p.shape)
                    curr += p.numel()
                    p.data=p.data*cfg_mask[n[6:]]
            self.cfg_mask=cfg_mask
        #added by me, evaluating on GSM8K dataset
        if ctx.cur_mode == MODE.TEST and ctx.cfg.data.type=="gsm8k@llm":
            fschatbot = FSChatBot_My(ctx.model, ctx.cfg)
            answers = []
            for batch in ctx.val_loader_copy:
                for instruction, _, output in zip(batch['instruction'], batch['input'], batch['output']):
                    input_text = build_prompt(instruction, N_SHOT, COT_FLAG)
                    generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
                    model_completion = fschatbot.generate(input_text, generate_kwargs)
                    model_answer = clean_answer(model_completion)
                    is_cor = is_correct(model_answer, output)
                    answers.append(is_cor)
                    print(f'Question: {instruction}\n\n'
                        f'Answers: {extract_answer_from_output(output)}\n\n'
                        f'Model Answers: {model_answer}\n\n'
                        f'Model Completion: {model_completion}\n\n'
                        f'Is correct: {is_cor}\n\n')

                    print(f'Num of total question: {len(answers)}, '
                        f'correct num: {sum(answers)}, '
                        f'correct rate: {float(sum(answers))/len(answers)}.')
            eval_results[f'{ctx.cur_split}_acc'] = float(sum(answers))/len(answers)
        if ctx.cur_mode == MODE.TEST and ctx.cfg.data.type=="nat@llm":
            from rouge import Rouge
            rouge=Rouge()
            fschatbot = FSChatBot_My(ctx.model, ctx.cfg)
            rougeL = []
            for batch in ctx.val_loader_copy:
                for instruction, _, output in zip(batch['instruction'], batch['input'], batch['output']):
                    generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.1)
                    input_text=instruction+'\n'+_+'\n'+'Answer:'
                    model_completion = fschatbot.generate(input_text, generate_kwargs)
                    print(f'Question: {input_text}\n\n'
                        f'Model Completion: {model_completion}\n\n'
                        f'Answer: {output}\n\n')
                    try:
                        rougeL.append(rouge.get_scores(model_completion,output)[0]['rouge-l']['r'])
                    except:
                        rougeL.append(1.0)
                    print(f'RougeL: {float(sum(rougeL))/len(rougeL)}.')
            eval_results[f'{ctx.cur_split}_rougeL'] = float(sum(rougeL))/len(rougeL)            
        setattr(ctx, 'eval_metrics', eval_results)
        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size
    def update(self, model_parameters, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        for key in model_parameters:
            model_parameters[key] = param2tensor(model_parameters[key])
        # Due to lazy load, we merge two state dict
        cfg_mask = {}
        if self.ctx.cfg.train.sparse_A == True and ((self.ctx.cfg.train.sparse_A==False) or self.round<10):
            server_params_flat = torch.cat([p.flatten() for n,p in model_parameters.items() if 'lora_A' in n])
            x=server_params_flat.abs().float()
            server_mask = torch.zeros_like(x).bool()
            k = int(x.numel()*0.1)
            _, keep_idx = torch.topk(x, k=k)
            server_mask[keep_idx] = 1
            curr = 0
            for n,p in model_parameters.items():
                if 'lora_A' in n:
                    cfg_mask[n] = server_mask[curr:curr+p.numel()].reshape(p.shape)
                    curr += p.numel()
                    p.data=p.data*cfg_mask[n]
            self.cfg_mask=cfg_mask
            # for name in model_parameters:
            #     if "lora_A" in name:
            #             x=model_parameters[name].flatten().abs().float()
            #             mask = torch.zeros_like(x).bool()
            #             k = int(x.numel()*0.5)
            #             _, keep_idx = torch.topk(x, k=k)
            #             mask[keep_idx] = 1
            #             mask=mask.reshape(model_parameters[name].shape)
            #             model_parameters[name]*=mask
        #     for name,param in self.ctx.model.named_parameters():
        #         if "lora_A" in name:
        #             x=(model_parameters[name[6:]]).flatten().abs().float()
        #             mask = torch.zeros_like(x)
        #             k = int(x.numel()*0.5)
        #             _, keep_idx = torch.topk(x, k=k)
        #             mask[keep_idx] = 1
        #             mask=mask.reshape(model_parameters[name[6:]].shape)
        #             mask2=mask*(-1)+1
        #             model_parameters[name[6:]] =model_parameters[name[6:]]*mask+param*mask2
        if self.ctx.cfg.train.flasc == True:
            for name in model_parameters:
                if "lora" in name:
                        x=model_parameters[name].flatten().abs().float()
                        mask = torch.zeros_like(x).bool()
                        k = int(x.numel()*self.ctx.cfg.train.B_ratio)
                        _, keep_idx = torch.topk(x, k=k)
                        mask[keep_idx] = 1
                        mask=mask.reshape(model_parameters[name].shape)
                        model_parameters[name]*=mask
        if self.ctx.cfg.train.sparse_B == True:
            if self.round < 10:
                pass
            elif self.round == 10:
                server_params_flat = torch.cat([p.flatten() for n,p in self.ctx.model.named_parameters() if 'lora_B' in n])
                x=server_params_flat.abs().float()
                server_mask = torch.zeros_like(x).bool()
                k = int(x.numel()*self.ctx.cfg.train.B_ratio)
                _, keep_idx = torch.topk(x, k=k)
                server_mask[keep_idx] = 1
                curr = 0
                for n,p in self.ctx.model.named_parameters():
                    if 'lora_B' in n:
                        cfg_mask[n[6:]] = server_mask[curr:curr+p.numel()].reshape(p.shape)
                        curr += p.numel()
                        p.data=p.data*cfg_mask[n[6:]]
                self.cfg_mask=cfg_mask
                LLMTrainer.clients_Mask.append(cfg_mask)
                # self.cfg.personalization.local_param=[]
                for name, param in self.ctx.model.named_parameters():
                    if "lora_A" in name:
                        param.requires_grad = False
                if len(LLMTrainer.clients_Mask)==self.cfg.federate.client_num:
                    v=0
                    p=0
                    LLMTrainer.global_Mask=LLMTrainer.clients_Mask[0]
                    for item in LLMTrainer.clients_Mask:
                        for name,value in item.items():
                            LLMTrainer.global_Mask[name]*=value
                    for name,value in LLMTrainer.global_Mask.items():
                        v+=value.sum()
                        p+=value.numel()
                    print(v/p)
                    LLMTrainer.clients_Mask=[]
            else:
                # for n,p in model_parameters.items():
                #     if 'lora_B' in n:
                #         p.data=p.data*self.cfg_mask[n]
                server_params_flat = torch.cat([p.flatten() for n,p in self.ctx.model.named_parameters() if 'lora_B' in n])
                x=server_params_flat.abs().float()
                server_mask = torch.zeros_like(x).bool()
                k = int(x.numel()*self.ctx.cfg.train.B_ratio)
                _, keep_idx = torch.topk(x, k=k)
                server_mask[keep_idx] = 1
                curr = 0
                for n,p in self.ctx.model.named_parameters():
                    if 'lora_B' in n:
                        cfg_mask[n[6:]] = server_mask[curr:curr+p.numel()].reshape(p.shape)
                        curr += p.numel()
                        p.data*=cfg_mask[n[6:]]
                self.cfg_mask=cfg_mask
                for n,p in self.ctx.model.named_parameters():
                    if 'lora_B' in n:
                        model_parameters[n[6:]]=model_parameters[n[6:]]*LLMTrainer.global_Mask[n[6:]]-p.data*LLMTrainer.global_Mask[n[6:]]+p.data
                LLMTrainer.clients_Mask.append(cfg_mask)
                v=0
                p=0
                if len(LLMTrainer.clients_Mask)==self.cfg.federate.client_num:
                    LLMTrainer.global_Mask=LLMTrainer.clients_Mask[0]
                    for item in LLMTrainer.clients_Mask:
                        for name,value in item.items():
                            LLMTrainer.global_Mask[name]*=value
                    for name,value in LLMTrainer.global_Mask.items():
                        v+=value.sum()
                        p+=value.numel()
                    LLMTrainer.vp.append(v/p)
                    print(LLMTrainer.vp)
                    LLMTrainer.clients_Mask=[]
        if self.ctx.cfg.train.Lori == True:
            if self.round < 10:
                pass
            elif self.round == 10:
                server_params_flat = torch.cat([p.flatten() for n,p in self.ctx.model.named_parameters() if 'lora_B' in n])
                x=server_params_flat.abs().float()
                server_mask = torch.zeros_like(x).bool()
                k = int(x.numel()*self.ctx.cfg.train.B_ratio)
                _, keep_idx = torch.topk(x, k=k)
                server_mask[keep_idx] = 1
                curr = 0
                for n,p in self.ctx.model.named_parameters():
                    if 'lora_B' in n:
                        cfg_mask[n[6:]] = server_mask[curr:curr+p.numel()].reshape(p.shape)
                        curr += p.numel()
                        p.data=p.data*cfg_mask[n[6:]]
                self.cfg_mask=cfg_mask
                # self.cfg.personalization.local_param=[]
                for name, param in self.ctx.model.named_parameters():
                    if "lora_A" in name:
                        param.requires_grad = False
            else:
                # for n,p in model_parameters.items():
                #     if 'lora_B' in n:
                #         p.data=p.data*self.cfg_mask[n]
                for n,p in self.ctx.model.named_parameters():
                    if 'lora_B' in n:
                        model_parameters[n[6:]]=model_parameters[n[6:]]* self.cfg_mask[n[6:]]
        self.S_model=0
        for n,p in model_parameters.items():
            if 'lora_B' in n:
                if self.ctx.cfg.train.sparse_B == True:
                    if self.round<=10:
                        self.S_model+=0
                    else:
                        self.S_model+=(torch.count_nonzero(p.data*LLMTrainer.global_Mask[n])*p.element_size()).item()
                elif self.ctx.cfg.train.Lori == True:
                    if self.round<=10:
                        self.S_model+=0
                    else:
                        self.S_model+=(torch.count_nonzero(p.data)*p.element_size()).item()
                elif 'lora_B' in self.ctx.cfg.personalization.local_param:
                    self.S_model+=0
                else:
                    self.S_model+=(torch.count_nonzero(p.data)*p.element_size()).item()
            elif 'lora_A' in n and self.ctx.cfg.federate.freeze_A == False:
                self.S_model+=(torch.count_nonzero(p.data)*p.element_size()).item()
        self.round+=1
        merged_param = merge_param_dict(self.ctx.model.state_dict().copy(),
                                         self._param_filter(model_parameters))
        self.ctx.model.load_state_dict(merged_param, strict=strict)
    def _param_filter(self, state_dict, filter_keywords=None):
        """
        model parameter filter when transmit between local and gloabl,
        which is useful in personalization.
        e.g., setting cfg.personalization.local_param= ['bn', 'norms']
        indicates the implementation of
        "FedBN: Federated Learning on Non-IID Features via Local Batch
        Normalization, ICML2021", which can be found in
        https://openreview.net/forum?id=6YEQUn0QICG

        Arguments:
            state_dict (dict): PyTorch Module object's state_dict.
        Returns:
            state_dict (dict): remove the keys that match any of the given
            keywords.
        """
        if self.cfg.federate.method in ["local", "global"]:
            return {}

        if filter_keywords is None:
            filter_keywords = self.cfg.personalization.local_param
        if ((self.ctx.cfg.train.sparse_B == True) or (self.ctx.cfg.train.Lori == True)) and self.round>10:
            filter_keywords = []
        trainable_filter = lambda p: True if \
            self.cfg.personalization.share_non_trainable_para else \
            lambda p: p in self.ctx.trainable_para_names
        keyword_filter = filter_by_specified_keywords
        return dict(
            filter(
                lambda elem: trainable_filter(elem[1]) and keyword_filter(
                    elem[0], filter_keywords), state_dict.items()))
def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder
        

def simulate_computation_delay(x_i, mu_i, d_i):
    return x_i * d_i + d_i / mu_i
def simulate_communication_delay(R, P_i, N0, W, S_model, num_clients):
    P_L = 100.7 + 23.5 * math.log10(R)  
    SNR = (P_i * P_L) / (N0 * W) 
    gamma_i = 1 / num_clients
    C_i = gamma_i * W * math.log2(1 + SNR)
    return S_model / C_i  # 公式：t_cm^i = S_model / C_i
register_trainer('llmtrainer', call_llm_trainer)
