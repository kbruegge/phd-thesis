import numpy as np


def matrix_to_latex(m, kind='pmatrix', precision=None, force_scientific=False):
    '''
    convert numpy array to latex matrx to be used with siunitx
    '''
    if precision:
        num_options = f'round-mode=places,round-precision={precision}'
    else:
        num_options = ''

    if force_scientific:
        num_options += ',scientific-notation=true'
    m_string = f'\\begin{{{kind}}}'
    for row in m:
        number_strings = [f'\\num[{{{num_options}}}]{{{a}}}' for a in row]
        row_string = ' & '.join(number_strings)
        row_string += '\\\\ '
        m_string += row_string

    # remove trailing slashes
    m_string = m_string[:-3] 
    m_string += f' \\end{{{kind}}}'
    return m_string


def latex_table(rows, column_names):
    align = 'l '
    m_string = f'\\begin{{tabular}}{{{align*len(column_names)}}}'
    
    titles = [f'\\textbf{{{c}}}' for c in column_names]
    title_string = ' & '.join(titles) + '\\\\ '
    m_string += title_string
    for row in rows:
        row_string = ' & '.join(row)
        row_string += '\\\\ '
        m_string += row_string
    m_string = m_string[:-3] 
    m_string += f' \\end{{tabular}}'
    return m_string


def asym_number(data, percentiles=[16, 50, 84]):
    '''Create a latex string for a number with asymetric errors
    
    Parameters
    ----------
    data : array-like
        the data from wbhich the percentiles are calculated
    percentiles : list, optional
        which percentiles to calculatre, by default [5, 50, 95]
    
    Returns
    -------
    string
        a latex string
    '''
    l, m, u = np.percentile(data, q=percentiles).T
    l = l - m
    u = u - m
    s = f'$\\num{{{m:.2f}}}\substack{{ {u:+.2f} \\\ {l:+.2f} }}$'
    return s
