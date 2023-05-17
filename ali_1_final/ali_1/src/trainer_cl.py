# 导入库
from init import *

from model_cl import *

logger = transformers.logging.get_logger(__name__)

#----------------------------------------
# trainer
class CLTrainer(Trainer):

    # 初始化
    def __init__(
        self, args_training, model, tokenizer,
        dataset_train=None, dataset_val=None, data_collator=None,
        do_fgm=False
    ):
        super(CLTrainer, self).__init__(
            args=args_training, model=model, tokenizer=tokenizer,
            train_dataset=dataset_train, eval_dataset=dataset_val, data_collator=data_collator
        )

        if do_fgm:
            self.fgm = FGM(model)

        #----------
        self.do_fgm = do_fgm

        return

    #--------------------
    # 训练一步
    def training_step(self, model, inputs):

        # 放到对应 CUDA
        inputs = self._prepare_inputs(inputs)

        args = self.args
        do_fgm = self.do_fgm
        fgm = self.fgm

        gradient_accumulation_steps = args.gradient_accumulation_steps
        n_gpu = args.n_gpu

        #----------\
        # 前向与反向
        model.train()
        loss = self.compute_loss(model, inputs)
        if n_gpu > 1:
            loss = loss.mean()
        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps
        loss.backward()

        #----------
        # 对抗
        if do_fgm:
            emb_name = 'word_embeddings.weight'
            fgm.attack(emb_name)
            loss_adv = self.compute_loss(model, inputs)
            if n_gpu > 1:
                loss_adv = loss_adv.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss_adv /= gradient_accumulation_steps
            loss_adv.backward()
            fgm.restore(emb_name)

        return loss.detach()

    #--------------------
    # 验证
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):

        args = self.args

        eval_batch_size = args.eval_batch_size
        world_size = args.world_size

        #----------
        time_start = time.time()
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        #----------
        # 验证循环所需参数
        kwargs_eval_loop = {
            'dataloader': eval_dataloader,
            'description': 'Eval',
            'metric_key_prefix': metric_key_prefix
        }

        output = self.evaluation_loop(**kwargs_eval_loop)

        #----------
        total_batch_size = eval_batch_size * world_size
        metric_update = speed_metrics(metric_key_prefix, time_start, num_samples=output.num_samples,
            num_steps=math.ceil(output.num_samples / total_batch_size))
        output.metrics.update(metric_update)

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    #--------------------
    # 验证循环
    def evaluation_loop(
        self, dataloader, description,
        prediction_loss_only=None, ignore_keys=None, metric_key_prefix='eval'
    ):

        args = self.args
        model = self._wrap_model(self.model, training=False)

        eval_accumulation_steps = args.eval_accumulation_steps
        local_rank = args.local_rank

        #----------
        batch_size = dataloader.batch_size
        if local_rank == 0:
            logger.info(f"***** Running {description} *****")
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            logger.info(f"  Batch size = {batch_size}")

        #----------
        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        if self.args.past_index >= 0:
            self._past = None

        losses_host, preds_host, labels_host = None, None, None
        all_losses, all_preds, all_labels = None, None, None

        #----------
        observed_num_examples = 0
        all_z1 = []
        all_z2 = []
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            # 预测一步
            loss, logits, labels = self.prediction_step(model, inputs)
            all_z1.append(logits.z1.cpu())
            all_z2.append(logits.z2.cpu())

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # 梯度累积后，放到 CPU 聚集
            if eval_accumulation_steps is not None and (step + 1) % eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, '_past'):
            delattr(self, '_past')

        #----------
        num_samples = observed_num_examples
        metrics = self.compute_eval_metrics(model=model, query=all_z1, doc=all_z2)
        metrics = denumpify_detensorize(metrics)

        for key in list(metrics.keys()):
            if not key.startswith(f'{metric_key_prefix}_'):
                metrics[f'{metric_key_prefix}_{key}'] = metrics.pop(key)

        kwargs_eval_loop_output = {
            'predictions': all_preds,
            'label_ids': all_labels,
            'metrics': metrics,
            'num_samples': num_samples
        }

        return EvalLoopOutput(**kwargs_eval_loop_output)

    #--------------------
    # 验证评价
    def compute_eval_metrics(self, model, query, doc):

        query = torch.cat(query, axis=0)
        doc = torch.cat(doc, axis=0)

        cos_sim = model.sim(query.unsqueeze(1), doc.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        labels = labels.cpu().numpy()
        preds = torch.argmax(cos_sim, dim=-1).long().cpu().numpy()
        result = accuracy_score(labels, preds)
        result = {'acc': result, 'eval_loss': loss}

        return result

    #--------------------
    # 预测一步
    def prediction_step(
        self, model, inputs,
        prediction_loss_only=False, ignore_keys=None
    ):

        # 放到对应 CUDA
        inputs = self._prepare_inputs(inputs)
        labels = None

        with torch.no_grad():
            loss = None
            logits = model(**inputs)
            if self.args.past_index >= 0:
                self._past = logits[self.args.past_index - 1]

        return (loss, logits, labels)