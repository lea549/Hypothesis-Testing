import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np


def hyp_test_fn():
    print('Hypothesis Testing function inputs')
    print('-----------------------------------------')

    try:

        # getting the sample size from user
        sample_size = int(input('Enter the sample size:'))

        if sample_size <= 0:
            print('sample size has to be greater than 0')

        # if the sample size is less than 50, making sure that the population is neither highly skewed nor has outliers
        elif sample_size < 50:
            skewness = input("\nIs the population highly skewed? Enter 'y' for Yes and 'n' for No:")
            if skewness == 'y':
                print('A sample size of 50 or more is recommended for highly skewed population distribution.')
            elif skewness == 'n':
                outliers = input("\nDoes the population contain outliers? Enter 'y' for Yes and 'n' for No:")
                if outliers == 'y':
                    print('A sample size of 50 or more is recommended for a population distribution with outliers.')
                elif outliers == 'n':
                    return  hyp_test_z_or_t(sample_size)
                else:
                    print('Enter the input in the right format')
            else:
                print('Enter the input in the right format')

        else:
            return  hyp_test_z_or_t(sample_size)

    except ValueError:
        print('Enter the input in the right format')


'''
case i: if the sample size is less than or equal to 30, t distribution is used. 
case ii: if the sample size is greater than 30 and the population standard deviation is known, z distribution is used. 
case iii: if the sample size is greater than 30 and the population standard deviation is unknown, t distribution is used.'''


def hyp_test_z_or_t(sample_size):
    try:
        # getting the population and sample mean from user
        pop_mean = float(input('\nEnter the population mean:'))
        sample_mean = float(input('\nEnter the sample mean:'))

        # case i: if the sample size is less than 30, t distribution is used
        if sample_size < 30:
            return t_dist(pop_mean, sample_size, sample_mean)

        else:
            pop_sd_known = input('''\nIs the population standard deviation known? Enter 'y' for Yes and 'n' for No: 
**for 'n', t-statistic will be considered.  ''')

            # case ii: if the sample size is greater than 30 and the population standard deviation is known,
            # z distribution is used
            if pop_sd_known == 'y':
                pop_sd = float(input('\nEnter the population standard deviation:'))
                return z_dist(pop_mean, sample_size, sample_mean, pop_sd)

            # case iii: if the sample size is greater than 30 and the population standard deviation is unknown,
            # t distribution is used
            elif pop_sd_known == 'n':
                return t_dist(pop_mean, sample_size, sample_mean)
            else:
                print('Enter the input in the right format')
    except ValueError:
        print('Enter the input in the right format')


def z_dist(pop_mean, sample_size, sample_mean, pop_sd):
    alpha = 0.05
    z_alpha = 1.68
    z_alpha_by_2 = 1.96
    z_or_t_dist = 0  # for print function to identify whether this is z or t distribution

    try:
        tail_count = int(input('\nIs this a one-tailed or two-tailed test? Enter 1 or 2:'))

        # making sure tail count value is either 1 or 2
        if tail_count == 1 or tail_count == 2:
            if tail_count == 1:
                tail_dir = input('\nRejection region is in the upper tail or lower tail, Enter u or l:')

                # if upper tail, set critical_value as z_alpha
                # if lower tail, set critical_value as -z_alpha
                if tail_dir == 'u' or tail_dir == 'U' or tail_dir == 'l' or tail_dir == 'L':
                    critical_value = z_alpha if tail_count == 1 and (tail_dir == 'u' or tail_dir == 'U') else -z_alpha

                    # calculating z_stats value
                    z_stats = round((sample_mean - pop_mean) / (pop_sd / sample_size ** 0.5), 3)

                    # consider taking another set of samples if z-stats==critical_value
                    if z_stats == critical_value:
                        print(
                            f'\nz-stats = z\u03B1 = {z_stats} in this case, consider taking another set of samples.\n')
                    else:
                        return print_results(sample_size, pop_mean, sample_mean, alpha, critical_value, z_stats,
                                             tail_count, tail_dir, z_or_t_dist)

                else:
                    print('Enter the input in the right format')
            else:
                critical_value = z_alpha_by_2  # for two-tailed test

                # calculating z_stats value
                z_stats = round((sample_mean - pop_mean) / (pop_sd / sample_size ** 0.5), 3)

                # consider taking another set of samples if z-stats==critical_value
                if z_stats == critical_value:
                    print(f'\nz-stats = z\u03B1/2 = {z_stats} in this case, consider taking another set of samples.\n')
                else:
                    return print_results(sample_size, pop_mean, sample_mean, alpha, critical_value, z_stats, tail_count,
                                         'both', z_or_t_dist)

        else:
            print('Tail count should be 1 or 2')

    except ValueError:
        print('Enter the input in the right format')
    except KeyError:
        print('This combination of confidence coefficient and tail count is not supported by this program')


def t_dist(pop_mean, sample_size, sample_mean):

    # T table values starts from degree of freedom (sample size - 1) = 1
    if sample_size < 2:
        print('Sample size has to be at least 2 for t-distribution')

    else:

        # loading t-table from t_table.csv
        # One Tail, Two Tails alpha values are in 0th and 1st row
        t_table = pd.read_csv('t_table.csv', index_col=0, header=None, names=[0, 1, 2, 3, 4, 5, 6])
        alpha = 0.05
        z_or_t_dist = 1  # for print function to identify whether this is z or t ditribution

        try:
            sample_sd = float(input('\nEnter the sample standard deviation:'))
            tail_count = int(input('\nIs this a one-tailed or two-tailed test? Enter 1 or 2:'))
            
            # making sure tail count value is either 1 or 2
            if tail_count == 1 or tail_count == 2:
                
                if tail_count == 1:
                    tail_dir = input('\nRejection region is in the upper tail or lower tail, Enter u or l:')
                
                    # if upper tail, set critical_value as t_alpha
                    # if lower tail, set critical_value as -t_alpha
                    if tail_dir == 'u' or tail_dir == 'U'  or tail_dir == 'l' or tail_dir == 'L':
                        
                        # fetching the corresponding t-value based on degree of freedom, tail count and confidence coefficient
                        # t_table.iloc[tail_count - 1, :] == alpha -> finding the right column, based on tail count and alpha
                        # t_table.iloc[row, column]
                        critical_value = t_table.iloc[sample_size + 1, t_table.columns[t_table.iloc[tail_count - 1, :] == alpha][0]]  
                        if tail_dir == 'l' or tail_dir == 'L':
                            critical_value = -critical_value
                            
                        # calculating t_stats value
                        t_stats = round((sample_mean-pop_mean) / (sample_sd/sample_size**0.5), 3)
                        
                        # consider taking another set of samples if t-stats==critical_value
                        if t_stats == critical_value:
                            print(f'\nt-stats = t\u03B1 = {t_stats} in this case, consider taking another set of samples.\n')
                        else:  
                            return print_results(sample_size, pop_mean, sample_mean, alpha, round(critical_value,4), t_stats, tail_count, tail_dir, z_or_t_dist)
               
                    else:
                        print('Enter the Rejection region input in the right format') 
                        
                else:
                    
                    # fetching the corresponding t-value based on degree of freedom, tail count and confidence coefficient
                    # t_table.iloc[tail_count - 1, :] == alpha -> finding the right column, based on tail count and alpha
                    # t_table.iloc[row, column]
                    critical_value = t_table.iloc[sample_size + 1, t_table.columns[t_table.iloc[tail_count - 1, :] == alpha][0]]  

                
                    # calculating t_stats value
                    t_stats = round((sample_mean-pop_mean) / (sample_sd/sample_size**0.5), 3)
                    
                    # consider taking another set of samples if t-stats==critical_value
                    if t_stats == critical_value:
                        print(f'\nt-stats = t\u03B1/2 = {t_stats} in this case, consider taking another set of samples.\n')
                    else:  
                        return print_results(sample_size, pop_mean, sample_mean, alpha, round(critical_value,4), t_stats, tail_count, 'both', z_or_t_dist)     
        
            else:
                print('Tail count should be either 1 or 2')    

        except ValueError:
            print('Enter the input in the right format')
        except IndexError:
            print('''Invalid combination of degree of freedom (sample size-1), tail count and confidence coefficient.''')


# function to print the results
def print_results(sample_size, pop_mean, sample_mean, alpha, critical_value, z_or_t_stats, tail_count, tail_dir,
                  z_or_t_dist):
    dist = 'z' if z_or_t_dist == 0 else 't'

    print('\nsample size:', sample_size) if z_or_t_dist == 0 else print('\nsample size: ' + str(sample_size) + ', dof:',
                                                                        sample_size - 1)
    print('sample mean:', sample_mean)

    print('\nHypothesis Test results')
    print('-------------------------------------')
    if tail_count == 1 and (tail_dir == 'l' or tail_dir == 'L'):
        print('type of hypothesis test: Lower-tailed test for Population Mean')
    elif tail_count == 1 and (tail_dir == 'u' or tail_dir == 'U'):
        print('type of hypothesis test: Upper-tailed test for Population Mean')
    else:
        print('type of hypothesis test: Two-tailed test for Population Mean')

    print('distribution considered: ' + dist + '-distribution')
    print('significance level, alpha:', alpha)
    print('critical value, ' + dist + '\u03B1/2:', critical_value) if tail_count == 2 else print(
        'critical value, ' + dist + '\u03B1:', critical_value)
    print('z-stats:', z_or_t_stats) if z_or_t_dist == 0 else print('t stats:', z_or_t_stats)

    if tail_count == 2:
        print(f'\nnull hypothesis: \u03bc = {pop_mean}')
        print(f'alternate hypothesis: \u03bc != {pop_mean}')
    else:
        if tail_dir == 'l' or tail_dir == 'L':
            print(f'\nnull hypothesis: \u03bc >= {pop_mean}')
            print(f'alternate hypothesis: \u03bc < {pop_mean}')
        else:
            print(f'\nnull hypothesis: \u03bc <= {pop_mean}')
            print(f'alternate hypothesis: \u03bc > {pop_mean}')

    # for pictorial representation
    plot_dist(dist, critical_value, z_or_t_stats, tail_count, tail_dir)
    print('\n***This image is purely for representation purpose only. Not to scale.***\n')

    if tail_count == 1 and tail_dir == 'l':
        if z_or_t_stats < critical_value:
            return f'since {dist}-stats < critical value, {dist}-stats is in the rejection region; hence we can reject null hypothesis.'
        else:
            return f'since {dist}-stats > critical value, {dist}-stats is in the acceptance region; hence we can accept null hypothesis.'
    elif tail_count == 1 and tail_dir == 'u':
        if z_or_t_stats > critical_value:
            return f'since {dist}-stats > critical value, {dist}-stats is in the rejection region; hence we can reject null hypothesis.'
        else:
            return f'since {dist}-stats < critical value, {dist}-stats is in the acceptance region; hence we can accept null hypothesis.'
    else:
        if z_or_t_stats < -critical_value:
            return f'since {dist}-stats < critical_value, {dist}-stats is in the rejection region, hence we can reject null hypothesis.'
        elif z_or_t_stats > critical_value:
            return f'since {dist}-stats > critical value, {dist}-stats is in the rejection region, hence we can reject null hypothesis.'
        else:
            return f'since {dist}-stats is in the acceptance region, we can accept null hypothesis.'


def plot_dist(dist, critical_value, z_or_t_stats, tail_count, tail_dir):
    plt.subplots(figsize =  (13, 5) if dist == 'z' else (10, 5))
    x = np.linspace(-9, 9, 200)
    y = stats.norm.pdf(x, 0, 1)
    plt.xlabel(dist + '-distribution')
    plt.yticks(ticks=[])
    plt.plot(x, y, color='black')

    if tail_count == 1 and tail_dir == 'l':
        if z_or_t_stats < critical_value:
            plt.xticks(ticks=[z_or_t_stats, critical_value, 0],
                       labels=[dist + '-stats=' + str(z_or_t_stats), dist + '\u03B1=' + str(critical_value), '\u03bc'],
                       rotation=90)
        elif z_or_t_stats < 0:
            plt.xticks(ticks=[critical_value, z_or_t_stats, 0],
                       labels=[dist + '\u03B1=' + str(critical_value), dist + '-stats=' + str(z_or_t_stats), '\u03bc'],
                       rotation=90)
        else:
            plt.xticks(ticks=[critical_value, 0, z_or_t_stats],
                       labels=[dist + '\u03B1=' + str(critical_value), '\u03bc', dist + '-stats=' + str(z_or_t_stats)],
                       rotation=90)
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi > critical_value for xi in x], color='#BDBDBD',
                         label='accceptance region')
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi < critical_value for xi in x], color='#6E6E6E',
                         label='rejection region')

    elif tail_count == 1 and tail_dir == 'u':
        if z_or_t_stats < 0:
            plt.xticks(ticks=[z_or_t_stats, 0, critical_value],
                       labels=[dist + '-stats=' + str(z_or_t_stats), '\u03bc', dist + '\u03B1=' + str(critical_value)],
                       rotation=90)
        elif z_or_t_stats < critical_value:
            plt.xticks(ticks=[0, z_or_t_stats, critical_value],
                       labels=['\u03bc', dist + '-stats=' + str(z_or_t_stats), dist + '\u03B1=' + str(critical_value)],
                       rotation=90)
        else:
            plt.xticks(ticks=[0, critical_value, z_or_t_stats],
                       labels=['\u03bc', dist + '\u03B1=' + str(critical_value), dist + '-stats=' + str(z_or_t_stats)],
                       rotation=90)
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi < critical_value for xi in x], color='#BDBDBD',
                         label='accceptance region')
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi > critical_value for xi in x], color='#6E6E6E',
                         label='rejection region')

    else:
        if z_or_t_stats < -critical_value:
            plt.xticks(ticks=[z_or_t_stats, -critical_value, 0, critical_value],
                       labels=[dist + '-stats=' + str(z_or_t_stats), dist + '\u03B1/2=-' + str(critical_value),
                               '\u03bc', dist + '\u03B1/2=' + str(critical_value)],
                       rotation=90)
        elif z_or_t_stats < 0:
            plt.xticks(ticks=[-critical_value, z_or_t_stats, 0, critical_value],
                       labels=[dist + '\u03B1/2=-' + str(critical_value), dist + '-stats=' + str(z_or_t_stats),
                               '\u03bc', dist + '\u03B1/2=' + str(critical_value)],
                       rotation=90)
        elif z_or_t_stats < critical_value:
            plt.xticks(ticks=[-critical_value, 0, z_or_t_stats, critical_value],
                       labels=[dist + '\u03B1/2=-' + str(critical_value), '\u03bc',
                               dist + '-stats=' + str(z_or_t_stats), dist + '\u03B1/2=' + str(critical_value)],
                       rotation=90)
        else:
            plt.xticks(ticks=[-critical_value, 0, critical_value, z_or_t_stats],
                       labels=[dist + '\u03B1/2=-' + str(critical_value), '\u03bc',
                               dist + '\u03B1/2=' + str(critical_value), dist + '-stats=' + str(z_or_t_stats)],
                       rotation=90)
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi > -critical_value and xi < critical_value for xi in x],
                         color='#BDBDBD', label='accceptance region')
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi < -critical_value for xi in x], color='#6E6E6E',
                         label='rejection region')
        plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=[xi > critical_value for xi in x], color='#6E6E6E')

    plt.axvline(z_or_t_stats, ymax=0.5)
    plt.legend()


# hyp_test_fn()