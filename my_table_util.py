import pandas as pd
import numpy as np

# 首先准备一个函数 

r"""
写一个函数 把表格转成 latex 表格格式

结果示例: 
\begin{table}[h!]
\centering
\caption{\textbf{收益率}的描述性统计}
\label{tab:desc_stats}
\begin{tabular}{p{0.5in}||p{1in}p{1in}p{1in}}
\toprule
& \textbf{STK1} & \textbf{STK2} & \textbf{STK3} \\
\midrule
\textbf{count} & 999 & 999 & 999 \\
\midrule
\textbf{mean} & 0.000703 & 0.000491 & 0.001121 \\
\textbf{std} & 0.018316 & 0.019481 & 0.024602 \\
\textbf{min} & -0.230239 & -0.142777 & -0.140714 \\
\textbf{25\%} & -0.006373 & -0.008691 & -0.010830 \\
\textbf{50\%} & 0.001000 & 0.000633 & 0.000590 \\
\textbf{75\%} & 0.010097 & 0.010550 & 0.014265 \\
max & 0.144286 & 0.148871 & 0.174189 \\
\bottomrule
\end{tabular}
\end{table}
"""

def simple_LaTeX_formating(df: pd.DataFrame, 
                           decimal: int =3, 
                           show_index: bool=True, 
                           bf_index: bool=True) -> str:
    """
    快速转 LaTeX tabular: 元素间 &，行尾 \\\\，无环境
    """
    # 1. 只处理数值 设置小数位
    df = df.apply(lambda s: s.round(decimal) if pd.api.types.is_numeric_dtype(s) else s)
    # txt = df.to_string(index=show_index, header=False)

    # 2. 表头列表
    # header = df.columns.tolist()
    # header = [' & '] + header + ['\\\\']

    # 3. 表身
    rows = df.to_numpy().tolist()

    # 4. 元素 -> str 防止非数值
    rows = [ [str(i) for i in row] for row in rows]

    # 5. 拼接 & 和 \\
    latex_rows =  [ ' & '.join(row) + ' \\\\' for row in rows]

    # 6. 加入索引
    if show_index:
        if bf_index:
            latex_rows = [ r'\textbf{' + str(df.index[i]) + r'}' + ' & ' + latex_rows[i] for i in range(len(latex_rows))]
        else: 
            latex_rows = [str(df.index[i]) + ' & ' + latex_rows[i] for i in range(len(latex_rows))]

    return '\n'.join( latex_rows )

# 调用
# print(simple_latex_formating(descriptive_statistics, 3))


def get_LaTeX_table(
        df: pd.DataFrame, 
        columns: list = None,
        decimal:int = 3,
        table = 'h!', col_style: str = 'c',
        caption: str = "", label: str = "", 
        show_index: bool = True, centering: bool = True, 
        textbf: dict ={'columns':False, 'index':False, 'caption':False}
) -> None: 
    """
    把表格转成 latex 表格格式
    ---
    参数:
    df: pandas.DataFrame
        需要转成 latex 表格格式的数据
    decimal: int
        小数位数
    table: str
        表格的格式
    col_style: str
        列的格式
    caption: str
        表格的标题
    label: str
        表格的标签
    show_index: bool
        是否显示索引
    centering: bool
        是否居中
    textbf: dict
        是否加粗
    """

    # df = df.round(decimal) # 对字符串 会引起bug

    # 方式2：apply + 类型判断（更通用）
    # df = df.apply(lambda s: s.round(decimal) if pd.api.types.is_numeric_dtype(s) else s)

    print(r'\begin{table}[' + table + r']')

    # 居中
    if centering:
        print(r'\centering')


    # 添加 caption
    if textbf['caption']:
        print(r'\caption{'+ r'\textbf{' + caption + r'}}')
    else:
        print(r'\caption{'+ caption +r'}')


    # label
    print(r'\label{tab:'+ label +r'}')


    print(r'\begin{tabular}{' + col_style*(df.shape[1] + show_index) + r'}') 
    print(r'\toprule')


    # 表头 (columns) 处理
    cols = df.columns.tolist() if (columns is None) else columns

    if textbf['columns']:
        cols = [ r'\textbf{' + str(col) + r'}' for col in cols]
    print( ' & '*show_index + ' & '.join(cols) + r' \\')

    print(r'\midrule')

    
    print(
        simple_LaTeX_formating(
            df=df, 
            decimal=decimal, 
            show_index=show_index, 
            bf_index=textbf['columns']
        )
    )


    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def add_significance(df, p_values, decimal=3):
    """
    根据 p 值数组在回归结果表中添加显著性标记。

    参数:
    df (pd.DataFrame): 回归结果表，包含系数等信息。
    p_values (np.ndarray): p 值数组，形状与 df 相同。
    decimal (int, optional): 小数位数，默认为 3。

    返回:
    pd.DataFrame: 添加了显著性标记的回归结果表。
    """
    # 确保 p_values 是 numpy 数组
    p_values = np.array(p_values)
    
    # 检查形状是否匹配
    if p_values.shape != df.shape:
        raise ValueError("p_values 的形状必须与 df 相同")
    
    # 创建一个与 df 形状相同的空 DataFrame，用于存储显著性标记
    significance = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # 根据 p 值添加显著性标记
    significance[p_values < 0.1] = '*'
    significance[p_values < 0.05] = '**'
    significance[p_values < 0.01] = '***'
    
    # 将显著性标记添加到原始回归结果表中
    data = df.astype('float').round(decimal)
    result = data.astype(str) + significance
    
    return result



def group_result_formatting(data: pd.DataFrame, 
                    groupby:list[str] = ['Size', 'Inv'], 
                    target: str = 'SMB', 
                    decimal: int = 3) -> pd.DataFrame: 
    """
    根据两个值 如: Size 和 Inv 将回归结果分组并格式化。
    ---
    参数:
    data (pd.DataFrame): 回归结果表，包含系数等信息。
    groupby (list[str], optional): 分组依据的列名列表，默认为 ['Size', 'Inv']。
    target (str, optional): 目标变量列名，默认为 'SMB'。
    decimal (int, optional): 小数位数，默认为 3。
    """

    df = data.round(decimal)[[groupby[0], groupby[1], target]]

    num_1 = data[groupby[0]].value_counts().shape[0]
    num_2 = data[groupby[1]].value_counts().shape[0]

    res = pd.DataFrame("", index = range(1, num_1+1), columns = range(1, num_2+1))

    # for i in range(1, num_1+1):
    #     for j in range(1, num_2+1):
    #         res.iloc[i-1, j-1] = df[ df[groupby[0]] == i  ][ df[groupby[1]] == j ][target].values[0]

    df_group = df.groupby(groupby)

    for i in range(1, num_1+1):
        for j in range(1, num_2+1):
            res.iloc[i-1, j-1] = df_group.get_group((i,j))[target].values[0]

    return res


