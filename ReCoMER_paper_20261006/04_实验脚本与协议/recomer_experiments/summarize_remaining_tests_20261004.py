"""Summarize every frozen test/control, including adverse findings.

Paired intervals resample whole M3ED dialogues and average differences across
the same three saved MH/fuser seeds. The cRBEF branch is fixed. These intervals
describe test-dialogue uncertainty, not independent full-model training seeds.
"""
import json
from pathlib import Path
import sys
import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def mean_sd(v):
    return dict(mean=float(np.mean(v)), sample_sd=float(np.std(v,ddof=1)) if len(v)>1 else None,n=len(v))


def wf1(y,pred,weight):
    cm=np.bincount(y*7+pred,weights=weight,minlength=49).reshape(7,7)
    support=cm.sum(1);denom=cm.sum(0)+support
    f=np.divide(2*cm.diagonal(),denom,out=np.zeros(7),where=denom>0)
    return float((support*f).sum()/max(support.sum(),1))


def calibration(p,y):
    p=p.astype(np.float64);confidence=p.max(1);correct=p.argmax(1)==y
    rows=[];ece=0.0
    index=np.minimum((confidence*15).astype(np.int64),14)
    for k in range(15):
        keep=index==k
        if not keep.any():continue
        acc=float(correct[keep].mean());conf=float(confidence[keep].mean())
        ece+=float(keep.mean())*abs(acc-conf)
        rows.append(dict(bin=k,n=int(keep.sum()),accuracy=acc,confidence=conf))
    target=np.eye(7)[y]
    return dict(ECE15_equal_width=ece,multiclass_Brier_sum=float(((p-target)**2).sum(1).mean()),bins=rows,
                scope='post-hoc descriptive calibration; no metric-driven model selection')


def main(root):
    plan=read(root/'PLAN.json');seeds=plan['seeds'];report=dict(scope='FROZEN_OFFICIAL_TEST_AND_MATCHED_CONTROLS',
        complete=(root/'COMPLETE.json').exists(),status=read(root/'STATUS.json'),limitations=plan['limitations'])
    full=[read(root/'full_test'/str(s)/'METRICS.json') for s in seeds if (root/'full_test'/str(s)/'METRICS.json').exists()]
    report['full_test']={k:{m:mean_sd([v['metrics'][k][m] for v in full]) for m in ('weighted_f1','macro_f1','accuracy','nll')}
                         for k in full[0]['metrics']} if full else {}
    report['full_test_etas']={str(v['seed']):v['eta'] for v in full}
    if (root/'BASELINE_TEST_SUMMARY.json').exists():
        rows=read(root/'BASELINE_TEST_SUMMARY.json');baseline={}
        for d in sorted(set(v['dataset'] for v in rows)):
            baseline[d]={}
            for arm in ('mhnou_final','no_history_reference'):
                group=[v for v in rows if v['dataset']==d and v['arm']==arm]
                keys=('weighted_f1','macro_f1','accuracy') if group and group[0]['task']=='cls' else ('mae','ccc','pearson','acc2','f12')
                if group:baseline[d][arm]=dict(lr=group[0]['lr'],metrics={k:mean_sd([v['metrics'][k] for v in group]) for k in keys})
        report['baseline_test']=baseline
    ablations={}
    for arm in ['no_relative_C',*plan['ablations']]:
        rows=[read(p) for p in sorted((root/'ablations'/arm).glob('*/TEST_METRICS.json'))]
        ablations[arm]=dict(completed=len(rows),seeds=[v['seed'] for v in rows],
           full_WF1=mean_sd([v['metrics']['ReCoMER']['weighted_f1'] for v in rows]) if rows else None,
           MH_WF1=mean_sd([v['metrics']['MHnoU']['weighted_f1'] for v in rows]) if rows else None,
           eta={str(v['seed']):v.get('eta') for v in rows},
           failures=[str(p.relative_to(root)) for p in (root/'ablations'/arm).glob('*/FAILURE.json')])
    report['ablations']=ablations
    mech={}
    for p in sorted((root/'mechanisms').glob('*/*/METRICS.json')):
        z=read(p);d=p.parent.parent.name
        mech.setdefault(d,[]).append(dict(seed=int(p.parent.name),alignment=z['contribution_rank_alignment'],
            admission=z['gate_counterfactual_counts'],clean_metrics=z['metrics'],off_metrics=z['metrics_history_off'],
            on_metrics=z['metrics_history_on'],groups=z['groups'],clean_ms=z['clean_ms_per_sample'],
            Shapley_ms=z['shapley_ms_per_sample'],parameters=z['parameters_total']))
    report['mechanisms']=mech
    report['feature_stress']={str(s):read(root/'full_test'/str(s)/'FEATURE_STRESS.json') for s in seeds
                             if (root/'full_test'/str(s)/'FEATURE_STRESS.json').exists()}
    report['objective_gradient_audit']=read(root/'OBJECTIVE_GRADIENT_AUDIT.json') if (root/'OBJECTIVE_GRADIENT_AUDIT.json').exists() else None
    for name in ('DATA_AUDIT','HARD_REJECTION_INFORMATION_FLOW_AUDIT','EXPERT_TO_MHNOU_INFORMATION_FLOW','CR_GATE_FORMULA_SENSITIVITY'):
        if (root/(name+'.json')).exists():report[name]=read(root/(name+'.json'))
    if (root/'full_feature_shapley/SUMMARY.json').exists():
        report['full_feature_intervention_shapley']=read(root/'full_feature_shapley/SUMMARY.json')
    # Paired, dialogue-clustered intervals. No seed or model is selected here.
    if len(full)==3:
        zs=[]
        for s in seeds:
            with np.load(root/'full_test'/str(s)/'PREDICTIONS.npz',allow_pickle=False) as z:zs.append(dict(z))
        ids=zs[0]['ids'];y=zs[0]['y_true'].astype(np.int64)
        assert all(np.array_equal(ids,z['ids']) and np.array_equal(y,z['y_true']) for z in zs)
        report['calibration']={role:[calibration(z['probabilities'] if role=='ReCoMER' else z[role],y) for z in zs]
                              for role in ('ReCoMER','MHnoU','cRBEF','equal_weight')}
        dialogue=np.asarray(['_'.join(v.split('_')[:-1]) for v in ids])
        unique,inverse=np.unique(dialogue,return_inverse=True)
        rng=np.random.default_rng(20261004)
        cluster_weights=np.array([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
        base_predictions=[z['probabilities'].argmax(1) for z in zs]
        pairs={name:[z[name].argmax(1) for z in zs] for name in ('MHnoU','cRBEF','equal_weight')}
        for arm in ablations:
            paths=[root/'ablations'/arm/str(s)/'TEST_PREDICTIONS.npz' for s in seeds]
            if all(p.exists() for p in paths):
                ps=[]
                for p in paths:
                    with np.load(p,allow_pickle=False) as z:
                        assert np.array_equal(ids,z['ids']);ps.append(z['probabilities'].argmax(1))
                pairs[arm]=ps
        comparisons={}
        for name,alternative in pairs.items():
            observed=[wf1(y,p,np.ones(len(y)))-wf1(y,q,np.ones(len(y))) for p,q in zip(base_predictions,alternative)]
            differences=[]
            for counts in cluster_weights:
                w=counts[inverse]
                differences.append(np.mean([wf1(y,p,w)-wf1(y,q,w) for p,q in zip(base_predictions,alternative)]))
            ci=np.quantile(differences,[.025,.975])
            comparisons[name]=dict(full_minus_control_WF1_percentage_points=100*float(np.mean(observed)),
                paired_seed_differences_percentage_points=[100*v for v in observed],
                bootstrap_95_percentile_interval_percentage_points=(100*ci).tolist(),resamples=1000,
                resampling_unit='whole dialogue',dialogues=len(unique),positive_seeds=sum(v>1e-12 for v in observed),
                scope='fixed trained models; cRBEF shared; not full-model seed-population uncertainty')
        report['paired_comparisons']=comparisons
        # Link labels/teacher to expert and outer decisions for descriptive analysis.
        linked=[]
        with np.load(root/'CR17_TEST_PREDICTIONS.npz',allow_pickle=False) as z:cr=dict(z)
        gate=cr['CR_gate'].reshape(-1);tav=cr['TAV'].argmax(1);cp=cr['CRBEF'].argmax(1)
        report['CR_gate_analysis']=dict(gate_mean=float(gate.mean()),gate_min=float(gate.min()),gate_max=float(gate.max()),
            TAV_wrong_CR_correct=int(((tav!=y)&(cp==y)).sum()),TAV_correct_CR_wrong=int(((tav==y)&(cp!=y)).sum()),
            CR_changes_TAV_prediction=int((tav!=cp).sum()),scope='frozen internal expert decisions, descriptive, not tuning')
        for s,z in zip(seeds,zs):
            p=root/'mechanisms/ReCoMER_MH_branch'/str(s)/'SAMPLE_DIAGNOSTICS.npz'
            if not p.exists():continue
            with np.load(p,allow_pickle=False) as d:arr=dict(d)
            assert np.array_equal(ids,arr['ids'])
            keep=(arr['modality_mask']>0).all(1);dominant=arr['shapley_raw'].argmax(1);groups={}
            for i,name in enumerate(('T','A','V')):
                mask=keep&(dominant==i)
                if not mask.any():continue
                groups[name]=dict(n=int(mask.sum()),mean_CR_gate=float(gate[mask].mean()),
                    MH_accuracy=float((z['MHnoU'][mask].argmax(1)==y[mask]).mean()),
                    CR_accuracy=float((z['cRBEF'][mask].argmax(1)==y[mask]).mean()),
                    full_accuracy=float((z['probabilities'][mask].argmax(1)==y[mask]).mean()),
                    weight_top1_teacher_agreement=float((arr['weights'][mask].argmax(1)==i).mean()))
            linked.append(dict(seed=s,teacher_dominant_groups=groups,scope='label-dependent measured Shapley grouping, analysis only'))
        report['linked_expert_mechanisms']=linked
    (root/'SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fmt=lambda v:f"{100*v['mean']:.2f} ± {100*v['sample_sd']:.2f}" if v['sample_sd'] is not None else f"{100*v['mean']:.2f}"
    lines=['# ReCoMER 剩余测试结果','',f"状态：{report['status']['phase']}；结果包含所有固定种子。",'',
           '## 当前完整模型：M3ED 正式测试','',
           '| 模型 | WF1（%，均值 ± 样本标准差） |','|---|---:|']
    for name,z in report['full_test'].items():lines.append(f"| {name} | {fmt(z['weighted_f1'])} |")
    lines+=['','cRBEF 为原始固定 seed17；三个种子仅改变 MHnoU 和外层融合器。不得将标准差解释成三个独立完整系统训练。',
            '测试模型及消融方案已冻结；此前其他版本已用过该测试集，因此不声称全新盲测。','',
            '## 完整模型与对照的配对差异','',
            '| 对照 | 完整模型减对照（WF1 百分点） | 对话分组 bootstrap 95% 区间 |','|---|---:|---:|']
    for name,z in report.get('paired_comparisons',{}).items():
        ci=z['bootstrap_95_percentile_interval_percentage_points']
        lines.append(f"| {name} | {z['full_minus_control_WF1_percentage_points']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines+=['','区间按完整对话重采样 1000 次，每次平均同一组三个固定种子的配对差值；描述测试对话不确定性。',
            '消融及分组比较为探索性分析，未作多重比较校正，不按显著性筛选结果。','',
            '## 重新训练的统一消融','',
            '| 控制项 | 已完成种子 | MHnoU 分支 WF1（%） | 完整融合 WF1（%） |','|---|---:|---:|---:|']
    for name,z in ablations.items():lines.append(f"| {name} | {z['completed']}/3 | {fmt(z['MH_WF1']) if z['MH_WF1'] else '尚未完成'} | {fmt(z['full_WF1']) if z['full_WF1'] else '尚未完成'} |")
    lines+=['','五个核心控制沿用同一源分组划分，重新训练四个内层折和最终分支，再拟合融合器；每个控制三种子。',
            'no_relative_C 只重新拟合融合器并将相对证据设零。闭环贡献输入反馈和历史残差反馈分别移除，不将任何一项称作整块闭环的纯消融。',
            '无 Shapley 监督控制仍保留教师测量以匹配算力，但 lambda_u=0；路由器因原设计的梯度分离失去该监督源。',
            'softmax 控制去掉居中有界映射，仍保留原有 eps_floor。运行时控制须用本次脚本重现，直接加载其 checkpoint 会恢复默认路径。','',
            '## 四数据集 MHnoU 测试','',
            '| 数据集 | MHnoU | 无历史/均匀权重参考 | 主指标 |','|---|---:|---:|---|']
    for name,z in report.get('baseline_test',{}).items():
        key='mae' if name=='MOSEI_full' else 'weighted_f1'
        cells=[]
        for arm in ('mhnou_final','no_history_reference'):
            if arm not in z:cells.append('未完成');continue
            v=z[arm]['metrics'][key]
            cells.append(f"{v['mean']:.4f} ± {v['sample_sd']:.4f}" if key=='mae' else fmt(v))
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {'MAE，越低越好' if key=='mae' else 'WF1（%）'} |")
    lines+=['','这里是独立的 MHnoU 训练与验证集选学习率，训练范围与完整 M3ED 模型不同；无历史参考同时使用均匀权重，不是单因素消融。',
            'MELD/IEMOCAP/MOSEI 尚无匹配 cRBEF 权重，不能把这些结果写成完整 ReCoMER 多数据集结果。MOSEI 为回归任务，单独讨论。','',
            '## 机制与计算量','',
            '逐样本实际 Shapley、预测贡献、模态权重、单模态输出、历史开关损失、弃权决策及 token mask 均保存在 mechanisms/。',
            '完整模型的 cRBEF/TAV/AV、内部专家门控和外层输入保存在 CR17_TEST_PREDICTIONS.npz 与 full_test/。SUMMARY.json 连接了专家门控与教师主导模态分组。',
            '参数量和特征输入推理吞吐保存在 full_test/*/COMPUTE.json；不含上游 Qwen/音频特征提取的总成本。',
            '六组特征置零和 mask 压力测试保存在 FEATURE_STRESS.json；不是重新编码原始缺失模态的标准基准。','',
            '| 核查对象 | 贡献排序 Top-1 一致率（均值） | 有历史样本中的硬拒绝数 / 每种子样本数 |','|---|---:|---:|']
    for name,group in mech.items():
        align=[z['alignment']['top1_agreement'] for z in group if z['alignment']]
        gates=[z['admission'] for z in group if z['admission']]
        align_text=f"{100*np.mean(align):.2f}%" if align else '不适用'
        gate_text='；'.join(f"{v['helpful_rejected']+v['harmful_rejected']} / {v['n']}" for v in gates) if gates else '不适用'
        lines.append(f'| {name} | {align_text} | {gate_text} |')
    lines+=['','当前完整模型的 MHnoU 分支：三个种子对全部 4,003 个有历史的测试样本均开启历史，未触发硬拒绝；空候选仍可通过注意力概率衰减残差，不能把二者混为一谈。',
            'EMPTY_CANDIDATE_AND_HISTORY_STATE.npz 补存空候选概率、历史状态特征和说话人标记。','',
            'full_feature_shapley/ 保存两条分支都纳入的特征干预贡献：逐一去掉当前及历史的 T/A/V，计算真实空输入基线上的三玩家 Shapley。文本干预将已提取的 Qwen 隐状态置零，并非重新编码缺失原始文本。与训练所用 MHnoU 加权 token 游戏分开解释。',
            'SUMMARY.json 补充 ECE（15 个等宽置信区间）及多类 Brier 分数；这些是事后描述性检查，未用于重新选择主模型或更换预定 WF1 主指标。','',
            '## 实现核查与仍缺少的证据','',
            'OBJECTIVE_GRADIENT_AUDIT.json 检查训练目标：闭环目标中的单模态预测头未获得梯度。它们用于融合的相对证据不能直接称为经过任务监督的单模态专家。',
            '这三个种子的单模态头权重与相同随机种子的初始化仅有约 1e-8 的 SWA 浮点差异；编码器本身仍通过联合任务目标训练。',
            'HARD_REJECTION_INFORMATION_FLOW_AUDIT.json：在训练集强制关闭历史残差后，改变历史特征仍通过路由器的历史状态输入造成约 1e-5 量级的 logit 差异；32 样本探针未改变类别。不能声称所有历史信息路径严格切断。',
            'EXPERT_TO_MHNOU_INFORMATION_FLOW.json：改变 cRBEF 输入只改变专家及外层融合输出，不改变 MHnoU 输出或相对证据。当前实现是类别概率融合，cRBEF 没有直接输入 MHnoU 的模态权重路由器。',
            '原始 cRBEF 的 Qwen 隐状态来自 recent4 对话提示，本身包含历史文本；MHnoU 的弃权门没有控制这条专家历史路径。因此只能讨论 MHnoU 历史残差的抑制，不能声称完整系统丢弃所有历史信息。',
            'CR_GATE_FORMULA_SENSITIVITY.json 保存六个冻结公式的事后敏感性对照，包含固定门控和去除先验校正；这些结果没有用于替换主模型。',
            'DATA_AUDIT.json：MELD 编号为分区内 diaN_uttM，跨分区重复名称不代表同样本。对应同名行的 T/A/V 三模态特征未发现完全相同，其余三数据集编号跨分区不重叠。该检查不能替代原始素材和特征生成谱系的完整审计。',
            '当前生产模型未因测试结果而改写；后续修复应作为新版本，在训练/验证数据上验证后再独立报告。',
            '仍缺：其他数据集的完整 cRBEF/ReCoMER 训练、可比的外部论文基线、包含上游编码器的端到端成本，以及完整上游不泄漏的 OOF 证据。','',
            '原始协议、权重哈希、代码快照和全部成功/失败命令见 PLAN.json、ATTEMPTS.json 和 logs/。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=report['complete'],phase=report['status']['phase'],full_test=report['full_test'],
                         ablation_completed={k:v['completed'] for k,v in ablations.items()})))


if __name__=='__main__':main(Path(sys.argv[1]))
