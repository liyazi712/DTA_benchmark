import copy
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
torch.set_num_threads(1)
import torch.nn.init as init
from gvp_transformer_pdbbind.dta import pdbbind_v2016, pdbbind_v2020, pdbbind_v2021_time, pdbbind_v2021_similarity
from gvp_transformer_pdbbind.model import DTAModel
from gvp_transformer_pdbbind.utils import Logger, Saver, EarlyStopping
from gvp_transformer_pdbbind.metrics import evaluation_metrics

class TripletLoss(nn.Module):
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(d_pos - d_neg + self.margin)
        return losses.mean()

def init_weights(m):
    if isinstance(m, nn.Linear):
        # 使用He初始化（Kaiming初始化）均匀分布初始化权重
        init.kaiming_uniform_(m.weight)
        # 如果有偏置项，初始化为0
        if m.bias is not None:
            init.zeros_(m.bias)

def _parallel_train_per_epoch(kwargs=None, n_epochs=None, eval_freq=None, monitoring_score='mse', loss_fn=None,
                              logger=None, save_checkpoint=False, test_after_train=True, output_dir=None):
    midx = kwargs['midx']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    best_model_state_dict = kwargs['best_model_state_dict']

    # 初始化早停机制
    stopper = EarlyStopping(patience=20, eval_freq=eval_freq, higher_better=False)

    model.train()
    model.apply(init_weights)
    for epoch in range(1, n_epochs + 1):
        feats_list = {
            'compound_feats': [],
            'protein_feats': [],
            'seq_feats': []
        }
        total_loss = 0
        contra_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            protein_seq = batch['full_seq'].to(device)
            pocket_seq = batch['poc_seq'].to(device)
            y = torch.tensor([float(item) for item in batch['y']]).to(device)
            optimizer.zero_grad()
            yh, compound_feats, protein_feats, seq_feats = model(xd, xp, protein_seq, pocket_seq)
            # print(compound_feats, protein_feats, seq_feats)
            loss = loss_fn(yh, y.view(-1, 1))
            # contra_loss = loss_fn(seq_feats, protein_feats)
            triplet_loss_fn = TripletLoss(margin=0.05) # 0.5
            triplet_loss = triplet_loss_fn(seq_feats, protein_feats, compound_feats)
            loss = loss + 10 * triplet_loss
            # loss = mse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            contra_loss += triplet_loss.item()
            feats_list['compound_feats'].append(compound_feats.detach().cpu().numpy())
            feats_list['protein_feats'].append(protein_feats.detach().cpu().numpy())
            feats_list['seq_feats'].append(seq_feats.detach().cpu().numpy())

        # 保存特征到文件
        # with open(f'./gvp_transformer_pdbbind/output/features/feats_epoch_{epoch}.pkl', 'wb') as f:
        #     pickle.dump(feats_list, f)

        train_loss = total_loss / len(train_loader)
        contra_loss = contra_loss / len(train_loader)

        if epoch % eval_freq == 0:
            val_results = _parallel_test(
                {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            is_best = stopper.update(val_results['metrics'][monitoring_score])
            if is_best:
                best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"M-{midx} E-{epoch}| Train Loss: {train_loss:.4f} | Triplet loss: {contra_loss:.4f} Valid Loss: {val_results['loss']:.4f} |"\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                )

        if stopper.early_stop:
            logger.info('Early stopping triggered at epoch {}'.format(epoch))
            break

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    if test_after_train:
        test_results = _parallel_test(
            {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
            loss_fn=loss_fn,
            test_tag=f"Model {midx}", print_log=True, logger=logger
        )
    if save_checkpoint:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state_dict, str(output_dir / f'checkpoint_{midx}.pt'))

    rets = dict(midx=midx, model=model)
    return rets


def _parallel_test(
    kwargs=None, loss_fn=None, 
    test_tag=None, print_log=False, logger=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yt, yp, total_loss = torch.Tensor(), torch.Tensor(), 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            protein_seq = batch['full_seq'].to(device)
            pocket_seq = batch['poc_seq'].to(device)
            y = torch.tensor([float(item) for item in batch['y']]).to(device)
            # yh = model(xd, xp, protein_seq, pocket_seq)
            # loss = loss_fn(yh, y.view(-1, 1))
            yh, compound_feats, protein_feats, seq_feats = model(xd, xp, protein_seq, pocket_seq)
            loss = loss_fn(yh, y.view(-1, 1))
            # contrative_loss = loss_fn(seq_feats, protein_feats)
            triplet_loss_fn = TripletLoss(margin=0.5)
            contrative_loss = triplet_loss_fn(seq_feats, protein_feats, compound_feats)
            loss = loss + 10 * contrative_loss
            total_loss += loss.item()
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=['mse', 'spearman', 'pearson']
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results


class DTAExperiment(object):
    def __init__(self,
        task=None,
        split_method=None,
        thre=None,
        prot_fc_dims=[1024, 128], drug_fc_dims=[1024, 128],
        mlp_dims=[1024, 512], mlp_dropout=0.25,
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
        n_ensembles=None, n_epochs=None, batch_size=None,
        lr=0.0001, onthefly=False, parallel=False,
        output_dir='./output', save_log=False, save_checkpoint=False, device=None
    ):
        self.saver = Saver(output_dir)
        self.output_dir = output_dir
        self.save_checkpoint = save_checkpoint
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)
        self.thre = thre
        print(f'thre: {self.thre}')
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        print(f"device: {self.device}")
        dataset_klass = {
            'pdbbind_v2020': pdbbind_v2020,
            'pdbbind_v2016': pdbbind_v2016,
            'pdbbind_v2021_time': pdbbind_v2021_time,
            'pdbbind_v2021_similarity': pdbbind_v2021_similarity
        }[task]

        self.dataset = dataset_klass(
            onthefly=onthefly,
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        self._task_data_df_split = None
        self._task_loader = None

        n_gpus = torch.cuda.device_count()
        # self.devices = [torch.device(f'cuda:{i % n_gpus}') for i in range(self.n_ensembles)]
        self.devices = [torch.device(f'cuda:{self.device}') for _ in range(self.n_ensembles)]
        # print(self.devices)
        self.model_config = dict(prot_fc_dims=prot_fc_dims, drug_fc_dims=drug_fc_dims,
                                 mlp_dims=mlp_dims,  mlp_dropout=mlp_dropout)
        self.build_model()
        self.criterion = F.mse_loss
        self.split_method = split_method
        self.logger.info(self.models[0])
        self.logger.info(self.optimizers[0])

    def build_model(self):

        # # 检查是否有多张GPU可用
        # model = DTAModel(**self.model_config)
        # if torch.cuda.device_count() > 1:
        #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
        #     # 将模型移动到第一个GPU上
        #     model = model.to('cuda:0')
        #     # 包装模型以使用多GPU
        #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        #
        #     # 创建模型ensemble
        #     self.models = [model for _ in range(self.n_ensembles)]
        # else:
        #     # 如果只有一张GPU或没有GPU，只使用一张GPU
        #     model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        #     # 创建模型ensemble
        #     self.models = [model for _ in range(self.n_ensembles)]
        #
        # # 为每个模型副本创建一个独立的优化器
        # self.optimizers = [optim.Adam(model.parameters(), lr=self.lr) for model in self.models]


        self.models = [DTAModel(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.lr) for model in self.models]

    def _get_data_loader(self, dataset, shuffle=False):
        return DataLoader(dataset=dataset,
                    batch_size=self.batch_size,
                    collate_fn=dataset.collate,
                    shuffle=shuffle,
                    drop_last=False)

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (split_data, split_df) = self.dataset.get_split()
            self._task_data_df_split = (split_data, split_df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]
    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def _format_predict_df(self, results, test_df=None, esb_yp=None):
        """
        results: dict with keys y_pred, y_true
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        # df.columns = ['num', 'pdb_id', 'smile', 'sequence', 'pocket', 'position', 'label']
        df.columns = ['pdb_id', 'smile', 'sequence', 'pocket', 'position', 'label', 'resolution', 'release_year']
        df['y_pred'] = results['y_pred']
        print(df)
        if esb_yp is not None:
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def train(self, n_epochs=None, patience=None, eval_freq=1, test_freq=None,
                monitoring_score='mse', rebuild_model=False, test_after_train=True):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        train_loader, valid_loader, test_loader = self.task_loader['train'], self.task_loader['valid'], \
        self.task_loader['test']

        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                    higher_better=(monitoring_score != 'mse'))
            rets = dict(
                midx = i + 1,
                model = self.models[i],
                optimizer = self.optimizers[i],
                device = self.devices[i],
                train_loader = train_loader,
                valid_loader = valid_loader,
                test_loader = test_loader,
                stopper = stp,
                best_model_state_dict = None,
            )
            rets_list.append(rets)

        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                n_epochs=n_epochs, eval_freq=eval_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion, logger=self.logger,
                test_after_train=test_after_train,
                save_checkpoint=self.save_checkpoint,
                output_dir=self.output_dir
            ) for i in range(self.n_ensembles))

        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']


    def test(self, test_model=None, test_df=None, save_prediction=False,
             save_df_name='prediction.tsv', test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        test_loader = self.task_loader['test']
        rets_list = []
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)

        esb_yp, esb_loss = None, 0
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }

        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['rmse', 'mae', 'spearman', 'pearson',]
        )
        results['metrics'] = eval_metrics
        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))

    def test_model_saved(self, model_file, test_df=None, save_prediction=False, save_df_name='prediction.tsv'):
        rets_list = []
        model_metrics_list = []
        esb_yp, esb_loss = None, 0
        metric_names = ['rmse', 'mae', 'spearman', 'pearson', 'rm2', 'ci']  # 定义评估指标名称
        for midx in range(self.n_ensembles):
            model = DTAModel(**self.model_config).to(self.devices[midx])
            model_path = f'./gvp_transformer_pdbbind/output/{model_file}/checkpoint_{midx + 1}.pt'
            model_state_dict = torch.load(model_path)
            model.load_state_dict(model_state_dict)
            test_loader = self.task_loader['test']
            rets = _parallel_test(
                kwargs={
                    'midx': midx + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[midx],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {midx + 1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)

            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else \
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))

            eval_metrics = evaluation_metrics(
                rets['y_true'], rets['y_pred'],
                eval_metrics=metric_names)

            model_metrics_list.append(pd.Series(eval_metrics, name=f'Model {midx + 1}'))
            esb_loss += rets['loss']

        mean_metrics = pd.DataFrame(model_metrics_list).mean()
        std_metrics = pd.DataFrame(model_metrics_list).std()
        model_metrics_list.append(pd.Series(mean_metrics, name='Mean'))
        model_metrics_list.append(pd.Series(std_metrics, name='Std'))
        y_true = rets['y_true']
        y_pred_avg = np.mean(esb_yp, axis=0)
        esb_loss /= self.n_ensembles
        results = {
            'y_true': y_true,
            'y_pred': y_pred_avg,
            'loss': esb_loss,
        }
        final_eval_metrics = evaluation_metrics(rets['y_true'], y_pred_avg, eval_metrics=metric_names)
        model_metrics_list.append(pd.Series(final_eval_metrics, name='Ensemble'))

        # 将所有的 Series 对象合并为一个 DataFrame
        all_metrics_df = pd.concat(model_metrics_list, axis=1)
        print(all_metrics_df)

        index = ['rmse', 'mae', 'spearman', 'pearson', 'rm2', 'ci']
        self.saver.save_df(all_metrics_df, 'all_model_metrics.tsv', float_format='%g', index=index)

        results['df'] = self._format_predict_df(results, test_df=test_df, esb_yp=esb_yp)

        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')