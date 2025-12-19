# CAPM模型在中国证券市场的有效性检验: 基于Black-Jensen-Scholes和Fama-MacBeth方法

*杉岛明菜official*

***

 **注意事项**:  
 > `data/preprocessed/daily_return.csv` 这个文件比较大, 占用云端的存储比较多, 或许**将来的某天**我会从仓库里删掉它, 然后换成一个网盘链接. 也可能我永远都不需要这么做. 无论如何, 有任何问题都欢迎提交 Issues 或者通过邮箱告知. 
 > 另外, `data` 路径下只保留了预处理过后的数据, raw数据并未存放, 预处理过程也未上传. 

 **本文参考**: 
 - Sharpe W F. Capital asset prices: A theory of market equilibrium under conditions of risk[J]. The journal of finance, 1964, 19(3): 425-442.
 - Black F, Jensen M C, Scholes M. The capital asset pricing model: Some empirical tests[J]. 1972.
 - Fama E F, MacBeth J D. Risk, return, and equilibrium: Empirical tests[J]. Journal of political economy, 1973, 81(3): 607-636.

## 实证检验方法

项目参照 Black‑Jensen‑Scholes（1972）与 Fama‑MacBeth（1973）两种经典计量框架, 基于1995‑2024年共三十年中国A股日度数据, 使用 `Python` 对 CAPM 在中国股市的有效性进行系统再检验. 

### 数据处理

我们对原始数据进行预处理, 最终得到了 30 年间个股的日超额收益率序列以及市场组合(使用上证综指收益率代理)的**日度超额收益率**序列. **所以, 下面讨论的都是超额收益率.**

### Black‑Jensen‑Scholes（1972）方法

`black-jensen-scholes.py` 中, 我们定义了 `CAPMBlackJensenScholes` 类, 该类包含以下功能: 

1. **时间序列检验**: 使用前5年数据, 使用OLS回归估计 $\beta$ 值, 方程为 CAPM 模型的基本形式: 
$r_{it} - r_{ft} = \beta_{i}(r_{mt} - r_{ft}) + \varepsilon_{it} .$
由此将 $\hat{\beta}$ 最小的 10% 的股票归入 *组合1*, 依此类推, 得到 10 个投资组合. 使用第 6 年的数据计算 10 个组合各自的平均超额收益率. 按照这种方式滚动计算, 得到 25 年 10 个投资组合的平均超额收益率. 
最终, 利用这 25 年的日度数据, 对每一个组合都做一个时间序列回归: 
$r_{p_j,t} - r_{ft} = \beta_{j}(r_{mt} - r_{ft}) + \varepsilon_{p_j,t},\quad j = 1, \ldots, 10.$
估计出 10 个投资组合的 $\beta$ 值. 


1. **横截面检验**: 使用上一步中计算得到的10个组合的平均日度收益率和 $\beta$ 值, 进行横截面回归: 
$r_{p_j,t} - r_{ft} = \gamma_0 + \gamma_1 \beta_{p_j,t} + \varepsilon_{p_j,t}, \quad j = 1, \ldots, 10.$

1. **核心假设检验**: 对于任一个组合, 若符合 CAPM 模型, 则 $\gamma_0$ 不应显著非0, 而 $\gamma_1$ 应显著为正. 

### Fama‑MacBeth（1973）方法

`fama-macbeth.py` 中, 我们定义了 `CAPMFamaMacbeth` 类, 该类包含以下功能: 

1. **组合形成**: 对于前 4 年数据, 使用 OLS 回归估计 $\beta$ 值, 和 BJS 方法类似, 最终得到20个组合. 
   
1. **初始估计**: 对于接下来 5 年数据, 在各组合内, 使用个股的日超额收益率数据直接对市场组合日超额收益率回归来分别估计个股的 $\beta$. 
将回归得到的个股 $\hat{\beta}_p$, $\hat{\beta}^2_p$ 和 残差的标准差 $\hat{S}_{\varepsilon t}$ 进⾏简单平均, 得到各组合的⻉塔系数 $\hat{\beta}_p$, 贝塔系数平方 $\hat{\beta}_p^2$ 和标准差 $\hat{S}_{\varepsilon t}$. 

1. **滚动更新**: 每年都重新计算组合的 $\hat{\beta}_p$, $\hat{\beta}^2_p$ 和 $\hat{S}_{\varepsilon t}$. 比如, 这 3 年是 1935-1937, 则计算区间分别是1930-1935, 1930-1936, 1930-1937. 

1. **模型检验**: 对于接下来 4 年的数据进行回归: 
$r_{p,t} - r_{f,t} = \gamma_{0 t} + \gamma_1 \hat{\beta}_{p,t-1} + \gamma_2 \hat{\beta}^2_{p,t-1} + \gamma_3 \hat{S}_{\varepsilon p ,t-1} + \varepsilon_{p,t}.$

1. **滚动重复**: 把组合确定区间向后推 1 年, 重复之前的操作. 

## 程序说明

请首先安装 `Python`, 然后安装依赖包: 

```bash
pip install -r requirements.txt
```

然后执行下面的命令: 

```bash
python black-jensen-scholes.py
python fama-macbeth.py
```

此外, 也可以自行修改 `black-jensen-scholes.py` 和 `fama-macbeth.py` 中的 `start_year` 和 `end_year` 参数, 来指定检验的年份范围. 

## 项目结构

```apl
CAPM-A-Share-Empirical "项目仓库"
├─ preprocess.py "数据预处理代码, 未上传, 忽略即可"
├─ black-jensen-scholes.py "实现 Black-Jensen-Scholes 方法的代码"
├─ fama-macbeth.py "实现 Fama-MacBeth 方法的代码"
├─ my_table_util.py "用于生成 LaTeX 表格代码的辅助包, 见 Greenmilkvvv/econ-utils"
├─ conclusion-summary.ipynb "分析结果并进行一些可视化"
├─ data "数据文件夹"
│  └─ raw "存放原始数据, 未上传, 忽略即可"
│  ├─ preprocessed "预处理后的收益率数据"
│  │  ├─ daily_return.csv "股票日度收益率"
│  │  ├─ market_yield.csv "市场组合收益率"
│  │  └─ risk_free_rate.csv "无风险利率"
└─ result "存放结果文件"
   ├─ black-jensen-scholes "Black-Jensen-Scholes 方法结果"
   │  └─ black_jensen_scholes_results.pkl "Black-Jensen-Scholes 方法结果文件"
   ├─ fama-macbeth "Fama-MacBeth 方法结果"
   │  └─ fama_macbeth_results.pkl "Fama-MacBeth 方法结果文件"
   └─ conclusion "可视化结果"
      ├─ CAPM_visualization.pdf "CAPM 模型可视化"
      ├─ 上市公司数量.pdf "上市公司数量的走势"
      ├─ 上市公司数量与上证指数.pdf "合并呈现上市公司数量、指数走势以及各轮行情"
      ├─ 全样本SML.pdf "全样本使用 BJS 回归得到的 SML 线"
      └─ 子样本SML.pdf "四个子样本使用 BJS 回归得到的 SML 线"

```

## 联系我们

如有任何问题, 或者需要数据, 欢迎通过 (greenmilkvvv@outlook.com) 告知. 

***

谢谢你们能看到这里. 