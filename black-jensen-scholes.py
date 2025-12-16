# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, List, Dict

import warnings
warnings.filterwarnings('ignore')

import pickle


# %%
class CAPMBlackJensenScholes:
    """
    Black-Jensen-Scholes CAPM实证检验实现
    使用日度A股数据进行复现
    """
    
    def __init__(self, n_groups: int = 10, formation_period_years: int = 5, 
                 testing_period_years: int = 1):
        """
        初始化
        
        Parameters:
        -----------
        n_groups : int, default=10
            分组数量
        formation_period_years : int, default=5  
            分组形成期年数
        testing_period_years : int, default=1
            测试期年数
        """
        self.n_groups = n_groups
        self.formation_period_years = formation_period_years
        self.testing_period_years = testing_period_years
        self.portfolio_betas = {}
        self.portfolio_returns = {}
        self.time_series_results = {}
        self.cross_section_results = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        """
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 添加年份列，便于按年分组
        data['year'] = data['date'].dt.year
        
        # 检查数据完整性
        print(f"数据时间范围: {data['date'].min()} 到 {data['date'].max()}")
        print(f"数据年份范围: {data['year'].min()} 到 {data['year'].max()}")
        print(f"股票数量: {data['code'].nunique()}")
        print(f"总观测值: {len(data)}")
        
        return data
    
    def get_year_date_range(self, data: pd.DataFrame, year: int) -> Tuple[str, str]:
        """
        获取指定年份的实际交易日范围
        """
        year_data = data[data['year'] == year]
        if year_data.empty:
            return None, None
        
        start_date = year_data['date'].min()
        end_date = year_data['date'].max()
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def estimate_stock_beta(self, stock_data: pd.DataFrame) -> float:
        """
        估计单只股票的Beta系数
        """
        if len(stock_data) < 60:
            return np.nan
            
        try:
            clean_data = stock_data.dropna(subset=['return', 'mkt_return'])
            if len(clean_data) < 60:
                return np.nan
                
            X = sm.add_constant(clean_data['mkt_return'])
            y = clean_data['return']
            
            model = sm.OLS(y, X).fit()
            return model.params[1]
        except Exception as e:
            return np.nan
    
    def form_portfolios(self, data: pd.DataFrame, formation_start_year: int, formation_end_year: int) -> Dict[str, List[str]]:
        """
        根据Beta值形成投资组合 - 按年份选择
        """
        formation_data = data[
            (data['year'] >= formation_start_year) & 
            (data['year'] <= formation_end_year)
        ].copy()
        
        print(f"形成期: {formation_start_year}-{formation_end_year}年, "
              f"实际日期: {formation_data['date'].min().strftime('%Y-%m-%d')} 到 {formation_data['date'].max().strftime('%Y-%m-%d')}, "
              f"股票数: {formation_data['code'].nunique()}")
        
        # 估计每只股票的Beta
        beta_estimates = []
        
        for code in formation_data['code'].unique():
            stock_data = formation_data[formation_data['code'] == code]
            if len(stock_data) >= 60:
                beta = self.estimate_stock_beta(stock_data)
                if not np.isnan(beta):
                    beta_estimates.append({'code': code, 'beta': beta})
        
        print(f"成功估计Beta的股票数量: {len(beta_estimates)}")
        
        if len(beta_estimates) < self.n_groups:
            print(f"股票数量不足分组: {len(beta_estimates)} < {self.n_groups}")
            return {}
        
        beta_df = pd.DataFrame(beta_estimates)
        
        # 按Beta值排序并分组
        beta_df = beta_df.sort_values('beta')
        beta_df['portfolio'] = pd.qcut(beta_df['beta'], self.n_groups, 
                                      labels=[f'Portfolio_{i+1}' for i in range(self.n_groups)],
                                      duplicates='drop')
        
        portfolios = {}
        for portfolio in beta_df['portfolio'].unique():
            portfolios[str(portfolio)] = beta_df[beta_df['portfolio'] == portfolio]['code'].tolist()
        
        print(f"形成组合: {len(portfolios)} 个组合")
        return portfolios
    
    def calculate_portfolio_returns(self, data: pd.DataFrame, portfolios: Dict[str, List[str]], 
                                  test_start_year: int, test_end_year: int) -> pd.DataFrame:
        """
        计算投资组合收益率 - 按年份选择
        """
        test_data = data[
            (data['year'] >= test_start_year) & 
            (data['year'] <= test_end_year)
        ].copy()
        
        if test_data.empty:
            return pd.DataFrame()
        
        portfolio_returns = []
        
        for portfolio_id, stocks in portfolios.items():
            portfolio_data = test_data[test_data['code'].isin(stocks)]
            
            if portfolio_data.empty:
                continue
                
            # 计算等权组合收益率
            daily_returns = portfolio_data.groupby('date')['return'].mean().reset_index()
            if daily_returns.empty:
                continue
                
            daily_returns['portfolio'] = portfolio_id
            
            # 获取市场收益率
            market_data = test_data[['date', 'mkt_return']].drop_duplicates('date')
            portfolio_data_merged = pd.merge(daily_returns, market_data, on='date', how='left')
            
            portfolio_returns.append(portfolio_data_merged)
        
        if portfolio_returns:
            result = pd.concat(portfolio_returns, ignore_index=True)
            actual_start = result['date'].min().strftime('%Y-%m-%d')
            actual_end = result['date'].max().strftime('%Y-%m-%d')
            print(f"测试期 {test_start_year}-{test_end_year}年: 实际日期 {actual_start} 到 {actual_end}, "
                  f"{len(result['portfolio'].unique())} 个组合")
            return result
        else:
            print(f"测试期 {test_start_year}-{test_end_year}年: 无法计算任何组合收益率")
            return pd.DataFrame()
    
    def rolling_portfolio_formation(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        滚动形成投资组合 - 按年份选择周期
        """
        min_year = data['year'].min()
        max_year = data['year'].max()
        
        print(f"数据年份范围: {min_year} 到 {max_year}")
        print(f"形成期: {self.formation_period_years}年, 测试期: {self.testing_period_years}年")
        
        portfolio_returns_all = {}
        window_count = 0
        
        # 计算可用的滚动窗口
        start_formation_year = min_year
        end_formation_year = max_year - self.testing_period_years
        
        current_formation_start = start_formation_year
        
        while current_formation_start + self.formation_period_years - 1 <= end_formation_year:
            formation_start_year = current_formation_start
            formation_end_year = formation_start_year + self.formation_period_years - 1
            test_start_year = formation_end_year + 1
            test_end_year = test_start_year + self.testing_period_years - 1
            
            # 确保测试期不超过数据范围
            if test_end_year > max_year:
                break
            
            print(f"\n窗口 {window_count + 1}:")
            print(f"  形成期: {formation_start_year}-{formation_end_year}年")
            print(f"  测试期: {test_start_year}-{test_end_year}年")
            
            # 形成组合
            portfolios = self.form_portfolios(data, formation_start_year, formation_end_year)
            
            if portfolios:
                # 计算组合收益率
                portfolio_returns = self.calculate_portfolio_returns(
                    data, portfolios, test_start_year, test_end_year
                )
                
                if not portfolio_returns.empty:
                    key = f"{test_start_year}_{test_end_year}"
                    portfolio_returns_all[key] = portfolio_returns
                    window_count += 1
                    print(f"  成功处理该窗口")
                else:
                    print(f"  无法计算组合收益率")
            else:
                print(f"  无法形成有效组合")
            
            # 滚动到下一年
            current_formation_start += 1
        
        print(f"\n总共成功处理 {window_count} 个滚动窗口")
        
        # 显示所有窗口信息
        if portfolio_returns_all:
            print("\n所有处理窗口:")
            for key in sorted(portfolio_returns_all.keys()):
                test_data = portfolio_returns_all[key]
                actual_start = test_data['date'].min().strftime('%Y-%m-%d')
                actual_end = test_data['date'].max().strftime('%Y-%m-%d')
                print(f"  {key}年: 实际日期 {actual_start} 到 {actual_end}")
        
        return portfolio_returns_all
    
    def estimate_portfolio_betas_full_period(self, portfolio_returns_all: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        使用完整的样本数据估计投资组合的Beta值
        """
        if not portfolio_returns_all:
            return {}
            
        # 合并所有期的组合收益率数据
        combined_data = pd.concat(portfolio_returns_all.values(), ignore_index=True)
        
        print(f"合并后总数据量: {len(combined_data)} 行")
        print(f"组合数量: {combined_data['portfolio'].nunique()}")
        print(f"数据时间范围: {combined_data['date'].min()} 到 {combined_data['date'].max()}")
        
        portfolio_betas = {}
        
        for portfolio in combined_data['portfolio'].unique():
            portfolio_data = combined_data[combined_data['portfolio'] == portfolio]
            portfolio_data = portfolio_data.dropna(subset=['return', 'mkt_return'])
            
            if len(portfolio_data) < 100:
                print(f"组合 {portfolio}: 数据点不足 ({len(portfolio_data)})")
                continue
                
            try:
                X = sm.add_constant(portfolio_data['mkt_return'])
                y = portfolio_data['return']
                
                model = sm.OLS(y, X).fit()
                beta = model.params[1]
                portfolio_betas[portfolio] = beta
                print(f"组合 {portfolio}: Beta = {beta:.4f} (基于 {len(portfolio_data)} 个观测值)")
            except Exception as e:
                print(f"组合 {portfolio} Beta估计失败: {e}")
                continue
        
        return portfolio_betas
    
    def time_series_test_full_period(self, portfolio_returns_all: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        使用完整数据的时间序列回归检验
        """
        if not portfolio_returns_all:
            return {}
            
        combined_data = pd.concat(portfolio_returns_all.values(), ignore_index=True)
        results = {}
        
        for portfolio in combined_data['portfolio'].unique():
            portfolio_data = combined_data[combined_data['portfolio'] == portfolio]
            portfolio_data = portfolio_data.dropna(subset=['return', 'mkt_return'])
            
            if len(portfolio_data) < 100:
                continue
                
            try:
                X = sm.add_constant(portfolio_data['mkt_return'])
                y = portfolio_data['return']
                
                model = sm.OLS(y, X).fit()
                
                results[portfolio] = {
                    'alpha': model.params[0],
                    'alpha_tstat': model.tvalues[0],
                    'alpha_pvalue': model.pvalues[0],
                    'beta': model.params[1],
                    'beta_tstat': model.tvalues[1],
                    'beta_pvalue': model.pvalues[1],
                    'rsquared': model.rsquared,
                    'n_obs': len(portfolio_data),
                    'period': f"{portfolio_data['date'].min().strftime('%Y-%m-%d')} to {portfolio_data['date'].max().strftime('%Y-%m-%d')}"
                }
            except Exception as e:
                print(f"组合 {portfolio} 时间序列回归失败: {e}")
                continue
        
        return results
    
    def cross_sectional_test_full_period(self, portfolio_returns_all: Dict[str, pd.DataFrame], 
                                       portfolio_betas: Dict[str, float]) -> Dict[str, any]:
        """
        使用完整数据的横截面回归检验
        """
        if not portfolio_returns_all:
            return {}
            
        combined_data = pd.concat(portfolio_returns_all.values(), ignore_index=True)
        
        # 计算每个组合在整个期间的平均收益率
        portfolio_stats = combined_data.groupby('portfolio').agg({
            'return': 'mean',
            'mkt_return': 'mean'
        }).reset_index()
        
        # 添加Beta值
        portfolio_stats['beta'] = portfolio_stats['portfolio'].map(portfolio_betas)
        portfolio_stats = portfolio_stats.dropna()
        
        print(f"横截面回归可用组合数量: {len(portfolio_stats)}")
        print("\n组合统计信息:")
        for _, row in portfolio_stats.iterrows():
            print(f"  {row['portfolio']}: 平均收益率={row['return']:.6f}, Beta={row['beta']:.4f}")
        
        if len(portfolio_stats) < 3:
            print("横截面回归: 组合数量不足")
            return {}
        
        # 横截面回归
        try:
            X = sm.add_constant(portfolio_stats['beta'])
            y = portfolio_stats['return']
            
            model = sm.OLS(y, X).fit()
            
            print(f"\n横截面回归结果:")
            print(f"  截距 (Gamma_0): {model.params[0]:.6f} (t-stat: {model.tvalues[0]:.4f}, p-value: {model.pvalues[0]:.4f})")
            print(f"  斜率 (Gamma_1): {model.params[1]:.6f} (t-stat: {model.tvalues[1]:.4f}, p-value: {model.pvalues[1]:.4f})")
            print(f"  R-squared: {model.rsquared:.4f}")
            
            return {
                'gamma_0': model.params[0],
                'gamma_0_tstat': model.tvalues[0],
                'gamma_0_pvalue': model.pvalues[0],
                'gamma_1': model.params[1],
                'gamma_1_tstat': model.tvalues[1],
                'gamma_1_pvalue': model.pvalues[1],
                'rsquared': model.rsquared,
                'n_observations': len(portfolio_stats),
                'portfolio_stats': portfolio_stats,
                'model_summary': str(model.summary())
            }
        except Exception as e:
            print(f"横截面回归失败: {e}")
            return {}
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        执行完整的BJS检验流程 - 按年份选择周期
        """
        print("开始BJS CAPM检验 (按年份选择周期)...")
        print("=" * 50)
        
        # 数据预处理
        data = self.prepare_data(df)
        
        # 滚动形成投资组合
        print("\n1. 滚动形成投资组合...")
        portfolio_returns_all = self.rolling_portfolio_formation(data)

        ## 计算收益率
        combined_data = pd.concat(portfolio_returns_all.values(), ignore_index=True)
        # 计算每个组合在整个期间的平均收益率
        self.portfolio_stats_data = combined_data.groupby('portfolio').agg({
            'return': 'mean',
            'mkt_return': 'mean'
        }).reset_index()
        
        if not portfolio_returns_all:
            print("错误: 无法形成有效的投资组合")
            return
        
        print(f"\n成功收集 {len(portfolio_returns_all)} 个测试期的组合收益率数据")
        
        # 估计组合Beta值（使用完整数据）
        print("\n2. 估计投资组合Beta值 (基于完整样本)...")
        self.portfolio_betas = self.estimate_portfolio_betas_full_period(portfolio_returns_all)
        
        if not self.portfolio_betas:
            print("错误: 无法估计组合Beta值")
            return
        
        print(f"\n成功估计 {len(self.portfolio_betas)} 个组合的Beta值")
        
        # 时间序列检验（使用完整数据）
        print("\n3. 执行时间序列回归检验 (完整样本)...")
        self.time_series_results = self.time_series_test_full_period(portfolio_returns_all)
        
        print(f"成功完成 {len(self.time_series_results)} 个组合的时间序列检验")
        
        # 横截面检验（使用完整数据）
        print("\n4. 执行横截面回归检验...")
        self.cross_section_results = self.cross_sectional_test_full_period(
            portfolio_returns_all, self.portfolio_betas
        )
        
        print("\nBJS CAPM检验完成!")
    
    def summary(self, return_data: bool = True) -> Dict[str, any]:
        """输出检验结果摘要，并返回结构化数据
        
        Parameters:
        -----------
        return_data : bool, default=True
            是否返回结构化数据
            
        Returns:
        --------
        Dict[str, any]
            包含所有检验结果的结构化数据
        """
        print("\n" + "="*70)
        print("Black-Jensen-Scholes CAPM检验结果摘要 (按年份选择周期)")
        print("="*70)
        
        # 准备返回的数据结构
        result_data = {
            'portfolio_stats': self.portfolio_stats_data,
            'portfolio_betas': {},
            'time_series_results': {},
            'cross_section_results': {},
            'summary_statistics': {}
        }
        
        # 1. 组合Beta值
        print(f"\n投资组合数量: {len(self.portfolio_betas)}")
        result_data['portfolio_betas'] = self.portfolio_betas.copy()
        
        print("\n1. 组合Beta估计值 (完整样本):")
        beta_values = []
        for portfolio, beta in sorted(self.portfolio_betas.items()):
            beta_values.append(beta)
            print(f"   {portfolio}: {beta:.4f}")
            result_data['portfolio_betas'][portfolio] = beta
        
        if beta_values:
            beta_stats = {
                'min': float(min(beta_values)),
                'max': float(max(beta_values)),
                'mean': float(np.mean(beta_values)),
                'median': float(np.median(beta_values)),
                'std': float(np.std(beta_values))
            }
            result_data['summary_statistics']['beta_stats'] = beta_stats
            
            print(f"\n   Beta统计:")
            print(f"   范围: {beta_stats['min']:.4f} - {beta_stats['max']:.4f}")
            print(f"   均值: {beta_stats['mean']:.4f}")
            print(f"   中位数: {beta_stats['median']:.4f}")
            print(f"   标准差: {beta_stats['std']:.4f}")
        
        # 2. 时间序列回归结果
        print("\n2. 时间序列回归结果 (Alpha检验 - 完整样本):")
        print("   Portfolio |     Alpha     | t-stat | p-value | Significant | 数据期间")
        print("   " + "-"*80)
        
        alpha_significant_count = 0
        alpha_values = []
        alpha_tstats = []
        
        result_data['time_series_results'] = {}
        
        for portfolio, result in sorted(self.time_series_results.items()):
            alpha_sig = "是" if result['alpha_pvalue'] < 0.05 else "否"
            if alpha_sig == "是":
                alpha_significant_count += 1
            alpha_values.append(result['alpha'])
            alpha_tstats.append(result['alpha_tstat'])
            
            print(f"   {portfolio:10} | {result['alpha']:12.6f} | {result['alpha_tstat']:6.2f} | "
                f"{result['alpha_pvalue']:7.4f} | {alpha_sig:>11} | {result['period']}")
            
            # 存储时间序列结果
            result_data['time_series_results'][portfolio] = {
                'alpha': float(result['alpha']),
                'alpha_tstat': float(result['alpha_tstat']),
                'alpha_pvalue': float(result['alpha_pvalue']),
                'beta': float(result['beta']),
                'beta_tstat': float(result['beta_tstat']),
                'beta_pvalue': float(result['beta_pvalue']),
                'rsquared': float(result['rsquared']),
                'n_obs': int(result['n_obs']),
                'period': result['period'],
                'significant': bool(result['alpha_pvalue'] < 0.05)
            }
        
        if alpha_values:
            alpha_stats = {
                'min': float(min(alpha_values)),
                'max': float(max(alpha_values)),
                'mean': float(np.mean(alpha_values)),
                'median': float(np.median(alpha_values)),
                'std': float(np.std(alpha_values)),
                'significant_count': int(alpha_significant_count),
                'total_count': len(self.time_series_results),
                'significant_ratio': float(alpha_significant_count / len(self.time_series_results))
            }
            result_data['summary_statistics']['alpha_stats'] = alpha_stats
            
            print(f"\n   Alpha统计:")
            print(f"   显著不为零的组合数量: {alpha_significant_count}/{len(self.time_series_results)}")
            print(f"   显著比例: {alpha_stats['significant_ratio']:.2%}")
            print(f"   Alpha均值: {alpha_stats['mean']:.6f}")
            print(f"   Alpha范围: {alpha_stats['min']:.6f} - {alpha_stats['max']:.6f}")
        
        # 3. 横截面回归结果
        print("\n3. 横截面回归结果 (完整样本):")
        if self.cross_section_results:
            cs_result = self.cross_section_results
            
            # 存储横截面结果
            result_data['cross_section_results'] = {
                'gamma_0': float(cs_result['gamma_0']),
                'gamma_0_tstat': float(cs_result['gamma_0_tstat']),
                'gamma_0_pvalue': float(cs_result['gamma_0_pvalue']),
                'gamma_1': float(cs_result['gamma_1']),
                'gamma_1_tstat': float(cs_result['gamma_1_tstat']),
                'gamma_1_pvalue': float(cs_result['gamma_1_pvalue']),
                'rsquared': float(cs_result['rsquared']),
                'n_observations': int(cs_result['n_observations']),
                'portfolio_stats': cs_result['portfolio_stats'].to_dict('records')
            }
            
            print(f"\n   Gamma_0 (截距): {cs_result['gamma_0']:.6f}")
            print(f"     t-stat: {cs_result['gamma_0_tstat']:.4f}")
            print(f"     p-value: {cs_result['gamma_0_pvalue']:.4f}")
            print(f"     显著性: {'显著' if cs_result['gamma_0_pvalue'] < 0.05 else '不显著'}")
            
            print(f"\n   Gamma_1 (斜率): {cs_result['gamma_1']:.6f}")
            print(f"     t-stat: {cs_result['gamma_1_tstat']:.4f}")
            print(f"     p-value: {cs_result['gamma_1_pvalue']:.4f}")
            print(f"     显著性: {'显著' if cs_result['gamma_1_pvalue'] < 0.05 else '不显著'}")
            
            print(f"\n   R-squared: {cs_result['rsquared']:.4f}")
            print(f"   观测值数量: {cs_result['n_observations']}")
            
            # CAPM理论预测分析
            print(f"\n   CAPM理论预测检验:")
            gamma_0_test = "通过" if abs(cs_result['gamma_0']) < 0.001 and cs_result['gamma_0_pvalue'] > 0.05 else "未通过"
            gamma_1_test = "通过" if cs_result['gamma_1_pvalue'] < 0.05 and cs_result['gamma_1'] > 0 else "未通过"
            capm_valid = gamma_0_test == "通过" and gamma_1_test == "通过"
            
            capm_test_results = {
                'gamma_0_test': gamma_0_test,
                'gamma_1_test': gamma_1_test,
                'capm_valid': capm_valid
            }
            result_data['summary_statistics']['capm_test_results'] = capm_test_results
            
            print(f"   - Gamma_0 ≈ 0: {gamma_0_test}")
            print(f"   - Gamma_1 > 0且显著: {gamma_1_test}")
            print(f"   - 整体CAPM有效性: {'支持' if capm_valid else '不支持'}")
        else:
            print("   横截面回归结果不可用")
            result_data['cross_section_results'] = {}
        
        # 添加总体信息
        result_data['summary_statistics']['overall'] = {
            'n_portfolios': len(self.portfolio_betas),
            'n_time_series_tests': len(self.time_series_results),
            'has_cross_section': bool(self.cross_section_results)
        }
        
        if return_data:
            return result_data


# %%
# 使用按年份选择周期的BJS检验
# bjs_test = CAPMBlackJensenScholes(
#     n_groups=10, 
#     formation_period_years=5,  # 5年形成期
#     testing_period_years=1     # 1年测试期
# )

# # 调整一下数据，把时间向前调整，期待结果有不同
# data = df.copy()
# data = data[ data['date'] >= '1995-01-01' ]

# bjs_test.fit(data)
# BJS_all_res = bjs_test.summary()

# print(BJS_all_res.keys())

# print( 
#     BJS_all_res['portfolio_betas']
# )

# print( 
#     BJS_all_res['time_series_results']
# )

# print( 
#     BJS_all_res['cross_section_results']
# )

# print( 
#     BJS_all_res['summary_statistics']
# )

# %%
# data = df.copy()
# data = data[ '1995-01-01' <= data['date'] <= '2002-12-31' ]
# bjs_test.fit(data)
# BJS_95_02_res = bjs_test.summary()


# %%
def run_BJS_test( 
        df, 
        start_date = '1995-01-01',
        end_date = '2024-12-31',
        n_groups=10, 
        formation_period_years=5, 
        testing_period_years=1
):
    
    bjs_test = CAPMBlackJensenScholes( 
        n_groups=n_groups, 
        formation_period_years=formation_period_years,  # 5年形成期
        testing_period_years=testing_period_years     # 1年测试期
    )
    
    data = df.copy()
    data = data[ 
        (data['date'] >= start_date) &
        (data['date'] <= end_date)
    ]
    bjs_test.fit(data)
    res = bjs_test.summary()

    return res


# %%
# 主执行函数
if __name__ == "__main__":

    # 读取数据
    df_daily_return = pd.read_csv("data/preprocessed/daily_return.csv")
    df_risk_free = pd.read_csv("data/preprocessed/risk_free_rate.csv")
    df_market_return = pd.read_csv("data/preprocessed/market_yield.csv")

    df = df_daily_return.merge(df_risk_free, on='date', how='left')
    df = df.merge(df_market_return, on='date', how='left')

    df['return'] = df['r_i'] - df['r_f']
    df['mkt_return'] = df['r_m'] - df['r_f']
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['r_i', 'r_f', 'r_m'], axis=1, inplace=True)


    BJS_all_res = run_BJS_test( df, start_date = '1995-01-01', end_date = '2024-12-31' )
    BJS_95_03_res = run_BJS_test( df, start_date = '1995-01-01', end_date = '2003-12-31' )
    BJS_02_10_res = run_BJS_test( df, start_date = '2002-01-01', end_date = '2010-12-31' )
    BJS_09_17_res = run_BJS_test( df, start_date = '2009-01-01', end_date = '2017-12-31' )
    BJS_16_24y_res = run_BJS_test( df, start_date = '2016-01-01', end_date = '2024-12-31' )

    BJS_res = { 
        'all': BJS_all_res,
        '95-03': BJS_95_03_res,
        '02-10': BJS_02_10_res,
        '09-17': BJS_09_17_res,
        '16-24': BJS_16_24y_res
    }

    
    filename = "result/black-jensen-scholes/black_jensen_scholes_results.pkl"
    with open(filename, 'wb') as f:  # 'wb'表示二进制写入
        pickle.dump(BJS_res, f)

    print(f"结果已保存到: {filename}")

