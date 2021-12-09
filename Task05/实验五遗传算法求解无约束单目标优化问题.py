import numpy as np
import matplotlib.pyplot as plt


# def fun(population):
#     x1 = population[:, 0]
#     x2 = population[:, 1]
#     return 21.5 + x1 * np.sin(4*np.pi*x1) + x2 * np.sin(20*np.pi*x2)

def fun(x1, x2):
    return 21.5 + x1 * np.sin(4*np.pi*x1) + x2 * np.sin(20*np.pi*x2)

# 从-1到2，等差取100个
X1s = np.linspace(-2.9, 12, 100)
X2s = np.linspace(4.2, 5.7, 100)

np.random.seed(0)  # 令随机数种子=0，确保每次取得相同的随机数

# 初始化原始种群
population_x1 = np.random.uniform(-2.9, 12, 10)  # 在[-1,2)上以均匀分布生成10个浮点数，做为初始种群
population_x2 = np.random.uniform(4.2, 5.7, 10)
population = np.hstack((population_x1.reshape(10, 1), population_x2.reshape(10, 1)))
print(population)
for pop1, pop2, fit in zip(population_x1, population_x2, fun(population_x1, population_x2)):
    print("x1=%5.2f, x2=%5.2f, fit=%.2f" % (pop1, pop2, fit))

X1s, X2s = np.meshgrid(X1s, X2s)
ax = plt.axes(projection='3d')
ax.plot_surface(X1s, X2s, fun(X1s, X2s))
ax.plot(population_x1, population_x2, fun(population_x1, population_x2), '*')
plt.show()


def encode(population_x1, population_x2, minX1=-2.9, maxX1=12, minX2=4.2, maxX2=5.7, scale=2**18, binary_len=18):  # population必须为float类型，否则精度不能保证
    # 标准化，使所有数据位于0和1之间,乘以scale使得数据间距拉大以便用二进制表示
    normalized_data1 = (population_x1-minX1) / (maxX1-minX1) * scale
    normalized_data2 = (population_x2-minX2) / (maxX2-minX2) * scale
    # 转成二进制编码
    binary_data = np.array([np.binary_repr(x1, width=binary_len) + np.binary_repr(x2, width=binary_len)
                            for x1, x2 in zip(normalized_data1.astype(int), normalized_data2.astype(int))])
    # print(binary_data)
    return binary_data


chroms = encode(population_x1, population_x2)  # 染色体英文(chromosome)


for pop1, pop2, chrom, fit in zip(population_x1, population_x2, chroms, fun(population_x1, population_x2)):
    print("x1=%.2f, x2=%.2f, chrom=%s, fit=%.2f" % (pop1, pop2, chrom, fit))


def decode(popular_gene, minX1=-2.9, maxX1=12, minX2=4.2, maxX2=5.7, scale=2**18):  # 先把x从2进制转换为10进制，表示这是第几份
    # 乘以每份长度（长度/份数）,加上起点,最终将一个2进制数，转换为x轴坐标
    # 将gene分成x1,x2
    gene1 = []
    gene2 = []
    for gene in popular_gene:
        g1, g2 = gene[:18],gene[18:]
        gene1.append(g1)
        gene2.append(g2)

    x1 = np.array([(int(x, base=2)/scale*(maxX1-minX1))+minX1 for x in gene1])
    x2 = np.array([(int(x, base=2)/scale*(maxX2-minX2))+minX2 for x in gene2])
    return x1, x2

x1, x2 = decode(chroms)
fitness = fun(x1, x2)

# for pop, chrom, dechrom, fit in zip(population, chroms, decode(chroms), fitness):
#     print("x=%5.2f, chrom=%s, dechrom=%.2f, fit=%.2f" %
#           (pop, chrom, dechrom, fit))

fitness = fitness - fitness.min() + 0.000001  # 保证所有的都为正
print(fitness)


def Select_Crossover(chroms, fitness, prob=0.6):  # 选择和交叉
    probs = fitness/np.sum(fitness)  # 各个个体被选择的概率
    probs_cum = np.cumsum(probs)  # 概率累加分布

    each_rand = np.random.uniform(size=len(fitness))  # 得到10个随机数，0到1之间

    # 轮盘赌，根据随机概率选择出新的基因编码
    # 对于each_rand中的每个随机数，找到被轮盘赌中的那个染色体
    newX = np.array([chroms[np.where(probs_cum > rand)[0][0]]
                    for rand in each_rand])
    # 繁殖，随机配对（概率为0.6)
    # 6这个数字怎么来的，根据遗传算法，假设有10个数，交叉概率为0.6，0和1一组，2和3一组。。。8和9一组，每组扔一个0到1之间的数字
    # 这个数字小于0.6就交叉，则平均下来应有三组进行交叉，即6个染色体要进行交叉
    pairs = np.random.permutation(
        int(len(newX)*prob//2*2)).reshape(-1, 2)  # 产生6个随机数，乱排一下，分成二列
    center = len(newX[0])//2  # 交叉方法采用最简单的，中心交叉法
    for i, j in pairs:
        # 在中间位置交叉
        x, y = newX[i], newX[j]
        newX[i] = x[:center] + y[center:]  # newX的元素都是字符串，可以直接用+号拼接
        newX[j] = y[:center] + x[center:]
    return newX


chroms = Select_Crossover(chroms, fitness)

dechroms_x1, dechroms_x2 = decode(chroms)
fitness = fun(dechroms_x1, dechroms_x2)

# for gene, dec, fit in zip(chroms, dechroms, fitness):
#     print("chrom=%s, dec=%5.2f, fit=%.2f" % (gene, dec, fit))

# 对比一下选择和交叉之后的结果
# fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(14, 5))
# axs1.plot(Xs, fun(Xs))
# axs1.plot(population, fun(population), 'o')
# axs2.plot(Xs, fun(Xs))
# axs2.plot(dechroms, fitness, '*')
# plt.show()

X1s, X2s = np.meshgrid(X1s, X2s)
ax = plt.axes(projection='3d')
ax.plot_surface(X1s, X2s, fun(X1s, X2s))
ax.plot(population_x1, population_x2, fun(population_x1, population_x2), 'o')
plt.show()

# 输入一个原始种群1，输出一个变异种群2  函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型。

# 长度由18变成36了
def Mutate(chroms: np.array):
    prob = 0.3  # 变异的概率
    clen = len(chroms[0])  # chroms[0]="111101101 000010110"    字符串的长度=18
    m = {'0': '1', '1': '0'}  # m是一个字典，包含两对：第一对0是key而1是value；第二对1是key而0是value
    newchroms = []  # 存放变异后的新种群
    each_prob = np.random.uniform(size=len(chroms))  # 随机10个数

    for i, chrom in enumerate(chroms):  # enumerate的作用是整一个i出来
        if each_prob[i] < prob:  # 如果要进行变异(i的用处在这里)
            pos = np.random.randint(clen)  # 从18个位置随机中找一个位置，假设是7
            # 0~6保持不变，8~17保持不变，仅将7号翻转，即0改为1，1改为0。注意chrom中字符不是1就是0
            chrom = chrom[:pos] + m[chrom[pos]] + chrom[pos+1:]
        newchroms.append(chrom)  # 无论if是否成立，都在newchroms中增加chroms的这个元素
    return np.array(newchroms)  # 返回变异后的种群


newchroms = Mutate(chroms)


def DrawTwoChroms(chroms1, chroms2, fitfun):  # 画2幅图，左边是旧种群，右边是新种群，观察平行的两幅图可以看出有没有差异
    Xs = np.linspace(-1, 2, 100)
    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(14, 5))
    dechroms = decode(chroms1)
    fitness = fitfun(dechroms)
    axs1.plot(Xs, fitfun(Xs))
    axs1.plot(dechroms, fitness, 'o')

    dechroms = decode(chroms2)
    fitness = fitfun(dechroms)
    axs2.plot(Xs, fitfun(Xs))
    axs2.plot(dechroms, fitness, '*')
    plt.show()


# 对比一下变异前后的结果
# DrawTwoChroms(chroms, newchroms, fun)

# 上述代码只是执行了一轮，这里反复迭代
np.random.seed(0)  #
population_x1 = np.random.uniform(-2.9, 12, 100)  # 在[-1,2)上以均匀分布生成10个浮点数，做为初始种群
population_x2 = np.random.uniform(4.2, 5.7, 100)
population = np.hstack((population_x1.reshape(100, 1), population_x2.reshape(100, 1)))
chroms = encode(population_x1, population_x2)

for i in range(1000):
    x1, x2 = decode(chroms)
    fitness = fun(x1, x2)
    # print("fit(%d):"%(i+1) + str(np.max(fitness)))
    print("fit(%d):"%(i+1) + str(fitness))
    fitness = fitness - fitness.min() + 0.000001  # 保证所有的都为正
    newchroms = Mutate(Select_Crossover(chroms, fitness))
    # if i % 300 == 1:
    #     DrawTwoChroms(chroms, newchroms, fun)


    chroms = newchroms

# DrawTwoChroms(chroms, newchroms, fun)