# encoding=utf-8

if __name__ == '__main__':
    csv_file = 'tmp/detailed_results/overall_comparisons.csv'
    algorithms = 6
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        cnt = 1
        for k, line in enumerate(lines[2:-1]):
            line_seps = line.strip().split(',')
            # template = '\\begin{tabular}[c]{@{}l@{}}%s\\\\ (%s)\\end{tabular} &'
            template = '{} ({}) '
            # template = '{} &'
            res = ''
            for i in range(algorithms):
                if line_seps[2 + 6*i] == '':
                    break
                # res += template % (line_seps[2 + 6*i], line_seps[7 + i*6])
                res += template.format(line_seps[2 + 6*i], line_seps[7 + i*6])
                if i != algorithms - 1:
                    res += ' & '
                # res += template.format(line_seps[2 + 6*i])
            if len(res) != 0:
                print((cnt-1) % 10 + 1, '&', res, '\\\\ \\hline')
                cnt += 1
