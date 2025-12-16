# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
# from datetime import datetime, timedelta
from scipy import stats
# from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')

import pickle
# import hashlib

# %%

class CAPMFamaMacBeth:
    def __init__(self, df, trading_days_per_year=252):
        """
        13年数据的Fama-MacBeth检验
        流程: 4年分组 + 5年初始化 + 34年滚动更新
        
        Parameters:
        df: 包含日期(date)、股票代码(code)、收益率(return)、市场收益率(mkt_return)的面板数据
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['code', 'date'])
        self.trading_days_per_year = trading_days_per_year
        self.portfolio_assignments = None
        
    def estimate_beta_simple(self, returns, market_returns, min_obs=60):
        """
        Beta估计
        """
        if len(returns) < min_obs:
            return np.nan, np.nan
        
        min_len = min(len(returns), len(market_returns))
        returns = returns[:min_len]
        market_returns = market_returns[:min_len]
        
        try:
            X = sm.add_constant(market_returns)
            model = sm.OLS(returns, X).fit()
            
            beta = model.params[1]
            resid_risk = np.std(model.resid)
            
            # 暂时不需要
            # if abs(beta) > 5 or np.isnan(beta):
            #     return np.nan, np.nan
                
            return beta, resid_risk
        except:
            return np.nan, np.nan
    
    def portfolio_formation_4years(self, start_date, end_date, n_portfolios=20):
        """
        阶段1: 4年组合构建期
        """
        print("=" * 60)
        print("阶段1: 4年组合构建期")
        print("=" * 60)
        print(f"时间: {start_date} 到 {end_date}")
        
        period_data = self.df[
            (self.df['date'] >= start_date) & 
            (self.df['date'] <= end_date)
        ].copy()
        
        print(f"期间内观测数: {len(period_data):,}")
        print(f"股票数量: {period_data['code'].nunique()}")
        
        # 筛选有足够数据的股票（4年约1000个交易日，要求至少200个观测）
        stock_obs = period_data['code'].value_counts()
        sufficient_data_codes = stock_obs[stock_obs >= 200].index.tolist()
        print(f"拥有至少200个观测值的股票数量: {len(sufficient_data_codes)}")
        
        # 估计Beta
        beta_estimates = []
        for code in sufficient_data_codes:
            stock_data = period_data[period_data['code'] == code]
            beta, resid_risk = self.estimate_beta_simple(
                stock_data['return'].values, 
                stock_data['mkt_return'].values,
                min_obs=200
            )
            if not np.isnan(beta):
                beta_estimates.append({
                    'code': code, 
                    'beta': beta,
                    'resid_risk': resid_risk,
                    'n_obs': len(stock_data)
                })
        
        beta_df = pd.DataFrame(beta_estimates)
        print(f"成功估计Beta的股票数量: {len(beta_df)}")
        
        if len(beta_df) < n_portfolios:
            n_portfolios = max(10, len(beta_df) // 2)
            print(f"调整组合数量为: {n_portfolios}")
        
        # 按Beta排序分组
        beta_df = beta_df.sort_values('beta')
        beta_df['portfolio'] = pd.qcut(
            beta_df['beta'], n_portfolios, labels=False, duplicates='drop'
        )
        
        self.portfolio_assignments = beta_df[['code', 'portfolio']]
        self.n_portfolios = len(beta_df['portfolio'].unique())
        
        print(f"成功构建 {self.n_portfolios} 个投资组合")
        
        # 组合统计
        portfolio_stats = beta_df.groupby('portfolio')['beta'].agg(['count', 'mean', 'std']).round(4)
        print("\n组合Beta统计:")
        print(portfolio_stats)
        
        return self.portfolio_assignments
    
    def initial_estimation_5years(self, start_date, end_date):
        """
        阶段2: 5年初始估计期 - 重新估计组合风险特征
        """
        print("\n" + "=" * 60)
        print("阶段2: 5年初始估计期")
        print("=" * 60)
        print(f"时间: {start_date} 到 {end_date}")
        
        period_data = self.df[
            (self.df['date'] >= start_date) & 
            (self.df['date'] <= end_date)
        ].copy()
        
        # 合并组合信息
        merged_data = period_data.merge(self.portfolio_assignments, on='code', how='inner')
        
        # 估计每个组合的初始Beta和残差风险
        portfolio_initial_stats = []
        
        for portfolio in range(self.n_portfolios):
            port_data = merged_data[merged_data['portfolio'] == portfolio]
            
            port_betas = []
            port_resid_risks = []
            
            for code in port_data['code'].unique():
                stock_data = port_data[port_data['code'] == code]
                beta, resid_risk = self.estimate_beta_simple(
                    stock_data['return'].values,
                    stock_data['mkt_return'].values,
                    min_obs=100
                )
                if not np.isnan(beta):
                    port_betas.append(beta)
                    port_resid_risks.append(resid_risk)
            
            if port_betas:
                portfolio_initial_stats.append({
                    'portfolio': portfolio,
                    'beta_initial': np.mean(port_betas),
                    'resid_risk_initial': np.mean(port_resid_risks),
                    'n_stocks': len(port_betas)
                })
        
        self.initial_stats = pd.DataFrame(portfolio_initial_stats)
        print("初始风险估计完成")
        print(self.initial_stats.round(4))
        
        return self.initial_stats
    
    def update_portfolio_betas_yearly(self, estimation_end_date):
        """
        逐年更新组合Beta估计
        """
        # 使用截至estimation_end_date的所有数据
        update_data = self.df[self.df['date'] <= estimation_end_date].copy()
        merged_data = update_data.merge(self.portfolio_assignments, on='code', how='inner')
        
        updated_betas = []
        
        for portfolio in range(self.n_portfolios):
            port_data = merged_data[merged_data['portfolio'] == portfolio]
            
            port_betas = []
            port_resid_risks = []
            
            for code in port_data['code'].unique():
                stock_data = port_data[port_data['code'] == code]
                # 使用至少1年数据估计
                beta, resid_risk = self.estimate_beta_simple(
                    stock_data['return'].values,
                    stock_data['mkt_return'].values,
                    min_obs=100
                )
                if not np.isnan(beta):
                    port_betas.append(beta)
                    port_resid_risks.append(resid_risk)
            
            if port_betas:
                updated_betas.append({
                    'portfolio': portfolio,
                    'beta_current': np.mean(port_betas),
                    'resid_risk_current': np.mean(port_resid_risks),
                    'estimation_end_date': estimation_end_date,
                    'n_stocks': len(port_betas)
                })
        
        return pd.DataFrame(updated_betas)
    
    def calculate_portfolio_returns(self, date):
        """
        计算组合日收益率
        """
        daily_data = self.df[self.df['date'] == date].copy()
        if len(daily_data) == 0:
            return None
        
        merged_data = daily_data.merge(self.portfolio_assignments, on='code', how='inner')
        
        if len(merged_data) == 0:
            return None
        
        # 等权重组合收益率
        portfolio_returns = merged_data.groupby('portfolio')['return'].mean().reset_index()
        portfolio_returns.columns = ['portfolio', 'daily_return']
        portfolio_returns['date'] = date
        portfolio_returns['n_stocks'] = merged_data.groupby('portfolio')['code'].nunique().values
        
        return portfolio_returns
    
    def cross_sectional_regression(self, portfolio_returns, portfolio_betas):
        """
        横截面回归
        """
        if len(portfolio_returns) < 10:  # 至少10个组合
            return None
        
        regression_data = portfolio_returns.merge(portfolio_betas, on='portfolio')
        regression_data['beta_sq'] = regression_data['beta_current'] ** 2
        
        X = sm.add_constant(regression_data[['beta_current', 'beta_sq', 'resid_risk_current']])
        y = regression_data['daily_return']
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HC1')
            
            return {
                'date': portfolio_returns['date'].iloc[0],
                'gamma_0': model.params['const'],
                'gamma_1': model.params['beta_current'],
                'gamma_2': model.params['beta_sq'],
                'gamma_3': model.params['resid_risk_current'],
                'n_portfolios': len(regression_data),
                'r_squared': model.rsquared
            }
        except:
            return None
    
    def run_12year_analysis(self, total_start_date, total_end_date):
        """
        运行完整的13年分析
        
        Parameters:
        total_start_date: 13年数据的开始日期 (e.g., '2000-01-01')
        total_end_date: 13年数据的结束日期 (e.g., '2011-12-31')
        """
        print("=" * 70)
        print("开始13年Fama-MacBeth分析")
        print("流程: 4年分组 + 5年初始化 + 4年滚动检验")
        print("=" * 70)
        
        # 验证数据时间范围
        data_start = self.df['date'].min()
        data_end = self.df['date'].max()
        print(f"数据时间范围: {data_start} 到 {data_end}")
        print(f"要求时间范围: {total_start_date} 到 {total_end_date}")
        
        # if data_start > pd.to_datetime(total_start_date) or data_end < pd.to_datetime(total_end_date):
        #     raise ValueError("数据时间范围不足以覆盖要求的13年期间")
        
        # 阶段1: 4年组合构建期 (前4年)
        formation_start = total_start_date
        formation_end = pd.to_datetime(total_start_date) + pd.DateOffset(years=4) - pd.DateOffset(days=1)
        
        # 阶段2: 5年初始估计期 (第5-9年)
        initial_start = formation_end + pd.DateOffset(days=1)
        initial_end = initial_start + pd.DateOffset(years=5) - pd.DateOffset(days=1)
        
        # 阶段3: 4年模型检验期 (最后4年)
        testing_start = initial_end + pd.DateOffset(days=1)
        testing_end = total_end_date
        
        print(f"\n时间安排:")
        print(f"组合构建期: {formation_start} 到 {formation_end} (4年)")
        print(f"初始估计期: {initial_start} 到 {initial_end} (5年)") 
        print(f"模型检验期: {testing_start} 到 {testing_end} (4年)")
        
        # 执行三个阶段
        # 1. 组合构建
        self.portfolio_formation_4years(formation_start.strftime('%Y-%m-%d'), 
                                      formation_end.strftime('%Y-%m-%d'))
        
        # 2. 初始估计
        self.initial_estimation_5years(initial_start.strftime('%Y-%m-%d'), 
                                     initial_end.strftime('%Y-%m-%d'))
        
        # 3. 模型检验期 - 逐年滚动
        print("\n" + "=" * 60)
        print("阶段3: 4年模型检验期 - 逐年滚动更新")
        print("=" * 60)
        
        testing_dates = self.df[
            (self.df['date'] >= testing_start) & 
            (self.df['date'] <= testing_end)
        ]['date'].unique()
        testing_dates = sorted(testing_dates)
        
        print(f"检验期交易日数量: {len(testing_dates)}")
        
        # 按年分组进行滚动更新
        testing_years = pd.Series(testing_dates).dt.year.unique()
        print(f"检验期包含年份: {list(testing_years)}")
        
        all_daily_results = []
        current_betas = None
        
        for year in testing_years:
            print(f"\n处理年份: {year}")
            
            # 每年开始时更新Beta估计（使用截至上一年底的数据）
            year_start = pd.Timestamp(f'{year}-01-01')
            estimation_end = year_start - pd.DateOffset(days=1)
            
            current_betas = self.update_portfolio_betas_yearly(estimation_end)
            print(f"  使用截至 {estimation_end.strftime('%Y-%m-%d')} 的数据更新Beta")
            
            # 处理该年的每个交易日
            year_dates = [d for d in testing_dates if d.year == year]
            
            for i, current_date in enumerate(year_dates):
                if (i + 1) % 50 == 0:
                    print(f"  进度: {i+1}/{len(year_dates)}")
                
                portfolio_returns = self.calculate_portfolio_returns(current_date)
                if portfolio_returns is None or current_betas is None:
                    continue
                
                result = self.cross_sectional_regression(portfolio_returns, current_betas)
                if result is not None:
                    result['year'] = year
                    all_daily_results.append(result)
        
        if len(all_daily_results) == 0:
            raise ValueError("没有成功的回归结果")
        
        results_df = pd.DataFrame(all_daily_results)
        
        print(f"\n检验完成")
        print(f"总回归天数: {len(results_df)}")
        print(f"年均回归天数: {len(results_df) / len(testing_years):.0f}")
        
        return results_df
    
    def hypothesis_testing(self, results_df):
        """
        假设检验
        """
        print("\n" + "=" * 60)
        print("假设检验结果")
        print("=" * 60)
        
        n_obs = len(results_df)
        
        # 按论文的四个假设进行检验
        hypotheses = {
            'gamma_0': 'H4: E(gamma_0) = R_f',
            'gamma_1': 'H3: E(gamma_1) > 0 (正风险溢价)', 
            'gamma_2': 'H1: E(gamma_2) = 0 (线性关系)',
            'gamma_3': 'H2: E(gamma_3) = 0 (Beta是完整风险度量)'
        }
        
        test_results = {} # 初始化回归结果储存
        # regression_table = pd.DataFrame( 
        #     data = "", 
        #     index =['hypothesis', 'mean', 'std', 'se', 't_stat', 'p_value', 'significance', 'n_obs'], 
        #     columns = list(hypotheses.keys())
        # )
        
        for coef_name, hypothesis in hypotheses.items():
            if coef_name not in results_df.columns:
                continue
                
            coef_series = results_df[coef_name].dropna()
            n_valid = len(coef_series)
            
            if n_valid < 50:
                print(f"{coef_name}: 有效观测值不足 ({n_valid})")
                continue
            
            # 计算统计量
            mean_coef = coef_series.mean()
            std_coef = coef_series.std()
            se_coef = std_coef / np.sqrt(n_valid)
            t_stat = mean_coef / se_coef
            
            # 计算p值（根据假设类型）
            if coef_name == 'gamma_1':
                p_value = stats.t.sf(t_stat, n_valid-1) if t_stat >= 0 else stats.t.cdf(t_stat, n_valid-1)
            else:
                p_value = 2 * stats.t.sf(np.abs(t_stat), n_valid-1)
            
            # 显著性
            if p_value < 0.01:
                significance = '***'
            elif p_value < 0.05:
                significance = '**' 
            elif p_value < 0.1:
                significance = '*'
            else:
                significance = ''

            # 存储结果 (表格)
            # regression_table.loc[coef_name, coef_name] = hypothesis  
            
            # 打印结果
            test_results[coef_name] = {
                'hypothesis': hypothesis,
                'mean': mean_coef,
                'std': std_coef,
                'se': se_coef,
                't_stat': t_stat,
                'p_value': p_value,
                'significance': significance,
                'n_obs': n_valid
            }

            
            
            print(f"\n{coef_name}: {hypothesis}")
            print(f"  均值: {mean_coef:.6f} {significance}")
            print(f"  标准误: {se_coef:.6f}")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_value:.4f}")
            print(f"  有效观测: {n_valid}")
        
        return test_results
    

    def get_regression_summary_dataframe(self, results_df):
        """
        生成回归结果统计的DataFrame
        返回格式整齐的统计结果表格
        """
        # 如果传入的是字典，提取results_df
        if isinstance(results_df, dict) and 'results_df' in results_df:
            results_df = results_df['results_df']
        
        if results_df is None or len(results_df) == 0:
            return pd.DataFrame()
        
        # 准备统计指标
        stats_data = []
        
        # 对每个系数进行统计
        coef_columns = ['gamma_0', 'gamma_1', 'gamma_2', 'gamma_3']
        
        for coef_name in coef_columns:
            if coef_name not in results_df.columns:
                continue
                
            coef_series = results_df[coef_name].dropna()
            n_obs = len(coef_series)
            
            if n_obs < 10:
                continue
            
            # 基本统计量
            mean_val = coef_series.mean()
            std_val = coef_series.std()
            se_val = std_val / np.sqrt(n_obs)
            t_stat = mean_val / se_val if se_val != 0 else 0
            
            # 计算p值（根据系数类型）
            if coef_name == 'gamma_1':
                # H3: 单边检验 (正风险溢价)
                p_value = stats.t.sf(t_stat, n_obs-1) if t_stat >= 0 else stats.t.cdf(t_stat, n_obs-1)
                hypothesis = "E(γ₁) > 0"
            else:
                # H1, H2, H4: 双边检验
                p_value = 2 * stats.t.sf(np.abs(t_stat), n_obs-1)
                if coef_name == 'gamma_0':
                    hypothesis = "E(γ₀) = 0"
                elif coef_name == 'gamma_2':
                    hypothesis = "E(γ₂) = 0"
                elif coef_name == 'gamma_3':
                    hypothesis = "E(γ₃) = 0"
            
            # 显著性标记
            if p_value < 0.01:
                significance = '***'
            elif p_value < 0.05:
                significance = '**'
            elif p_value < 0.1:
                significance = '*'
            else:
                significance = ''
            
            stats_data.append({
                'Coefficient': coef_name,
                'Hypothesis': hypothesis,
                'Mean': mean_val,
                'Std_Error': se_val,
                't_Statistic': t_stat,
                'p_Value': p_value,
                'Significance': significance,
                'N_Observations': n_obs,
                'Mean_With_Significance': f"{mean_val:.6f}{significance}"
            })
        
        # 创建统计表格
        summary_df = pd.DataFrame(stats_data)
        
        # 添加R-squared统计
        if 'r_squared' in results_df.columns:
            r2_stats = {
                'Coefficient': 'R_Squared',
                'Hypothesis': '-',
                'Mean': results_df['r_squared'].mean(),
                'Std_Error': results_df['r_squared'].std() / np.sqrt(len(results_df)),
                't_Statistic': np.nan,
                'p_Value': np.nan,
                'Significance': '',
                'N_Observations': len(results_df['r_squared'].dropna()),
                'Mean_With_Significance': f"{results_df['r_squared'].mean():.6f}"
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([r2_stats])], ignore_index=True)
        
        # 添加组合数量统计
        if 'n_portfolios' in results_df.columns:
            n_portfolios_stats = {
                'Coefficient': 'N_Portfolios',
                'Hypothesis': '-', 
                'Mean': results_df['n_portfolios'].mean(),
                'Std_Error': results_df['n_portfolios'].std() / np.sqrt(len(results_df)),
                't_Statistic': np.nan,
                'p_Value': np.nan,
                'Significance': '',
                'N_Observations': len(results_df['n_portfolios'].dropna()),
                'Mean_With_Significance': f"{results_df['n_portfolios'].mean():.2f}"
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([n_portfolios_stats])], ignore_index=True)
        
        # 设置显示格式
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        return summary_df
    


# %%

def run_one_round_fama_macbeth(df, start_year=2000, end_year=2012):
    """
    运行完整的13年Fama-MacBeth分析
    
    Parameters:
    df: 包含date, code, return, mkt_return的数据
    start_year: 开始年份
    end_year: 结束年份
    """
    
    # 检查数据是否覆盖要求的时间范围
    data_start = df['date'].min()
    data_end = df['date'].max()
    
    required_start = f'{start_year}-01-01'
    required_end = f'{end_year}-12-31'
    
    print(f"要求时间范围: {required_start} 到 {required_end}")
    print(f"数据时间范围: {data_start} 到 {data_end}")
    
    if data_start > pd.to_datetime(required_start) or data_end < pd.to_datetime(required_end):
        print("警告: 数据时间范围可能不足")
    
    # 初始化13年分析器
    analyzer = CAPMFamaMacBeth(df)
    
    # 运行完整分析
    results_df = analyzer.run_12year_analysis(
        total_start_date=pd.to_datetime(required_start),
        total_end_date=pd.to_datetime(required_end)
    )
    
    # 假设检验
    test_results = analyzer.hypothesis_testing(results_df)
    
    print("\n 13年Fama-MacBeth分析完成!")
    
    return {
        'results_df': results_df,
        'test_results': test_results,
        'portfolio_assignments': analyzer.portfolio_assignments
    }


def run_multirounds_fama_macbeth(df: pd.DataFrame, year_lst: list[list[int, int]]):  
    """对于不同的年份组合进行不同的Fama-Macbeth分析"""

    all_results = [] # 初始化列表准备存放字典

    for i in range( len(year_lst) ): 
        print("="*20 + "第{}轮".format(i+1) + "="*20)

        analyzer = CAPMFamaMacBeth(df)
        results = run_one_round_fama_macbeth(df, year_lst[i][0], year_lst[i][1])
        results['summary_df'] = analyzer.get_regression_summary_dataframe(results)

        if i < len(year_lst)-1: 
            print('='*20 + '进入下一轮' + '='*20)
        else: 
            print('='*20 + '所有轮次完成' + '='*20)

        all_results.append(results)
    return all_results




# %%
# 主执行函数
if __name__ == "__main__":

    # 读取数据
    df_daily_return = pd.read_csv("data/preprocessed/daily_return.csv")
    df_risk_free = pd.read_csv("data/preprocessed/risk_free_rate.csv")
    df_market_return = pd.read_csv("data/preprocessed/market_yield.csv")
    # 数据合并
    df = df_daily_return.merge(df_risk_free, on='date', how='left')
    df = df.merge(df_market_return, on='date', how='left')
    df['return'] = df['r_i'] - df['r_f']
    df['mkt_return'] = df['r_m'] - df['r_f']
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['r_i', 'r_f', 'r_m'], axis=1, inplace=True)


    # 遍历所有年份组合
    year_lst = range(1995, 2024-12+1)
    year_lst = [[x, x+12] for x in year_lst]
    
    analyzer = CAPMFamaMacBeth(df)
    all_results = run_multirounds_fama_macbeth(df, year_lst)

    # 将结果保存在 pickle 当中
    filename = "result/fama-macbeth/fama_macbeth_results.pkl"
    with open(filename, 'wb') as f:  # 'wb'表示二进制写入
        pickle.dump(all_results, f)
    
    print(f"结果已保存到: {filename}")
    
