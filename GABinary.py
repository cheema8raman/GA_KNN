#استعاء المكتبات المستخدمة
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# انشاء خوارزمية الجينية
class BGA:
    # يتم استدعاء الدالة كلما تم انشاء عنصر من الفئة (الكلاس) الرئيسية
    def __init__(self, dimension=2, pop_size=50, pc=0.9, pm=0.005, bit_length=(5, 5),
                 low=(1, 1), hi=(20, 30), max_iter=100,  log=True):

        # المتغيرات
        self.pm = pm # نسبة الطفرة
        self.pc = pc # نسبة التهجين
        self.bit_length = bit_length # حجم الجينات
        self.chromosome_length = sum(bit_length)  # Li
        self.pop_size = pop_size  # حجم الجيل الواحد
        self.dimension = dimension  # الاقسام في داخل الكروموسوم

        # قيم المتغيرات التي تريد االعمل بها
        self.hi = hi # القيمة العليا
        self.low = low # القيمة السفلى
        self.log = log #
        self.max_iter = max_iter

        # دالة الاقوى
        # يتم تحديده من المستخدم
        self.fitness_function = None

        # لتحديد القيم للجيل
        self.population_fitness = None
        self.decoded_population = None

        # لتتبع المسار للقيم
        self.best_so_far = []
        self.average_fitness = []

        # توليد عدد عشوائي من الجيل
        self.population = np.random.randint(0, 2, (self.pop_size, self.chromosome_length))

    # تحديد الدالة التي سوف يتم استخدامها لحساب الحلول الافضل
    def set_fitness(self, fitness_func):
        self.fitness_function = fitness_func

    # دالة الانتخاب الجماعي
    def selection(self, N, population, probability):
        pool = []
        for i in range(N):
            # len(population) can also be self.pop_size
            index = np.random.choice(len(population), p=probability)
            pool.append(population[index])

        return np.array(pool)

    #دالة التهجين
    def crossover(self, pool):
        new_population = []
        for i in range(0, len(pool), 2):
            child_1 = pool[i]
            try:
                child_2 = pool[i + 1]
            except ValueError:
                print("القيم المدخلة غير صحيحة")

            # اختبار التهجين اذا نجحت النسبة
            if np.random.uniform() < self.pc:
                cut_point = np.random.randint(1, self.chromosome_length)
                temp = child_2[:cut_point].copy()
                child_2[:cut_point], child_1[:cut_point] = child_1[:cut_point], temp

            new_population.extend([child_1, child_2])

        return new_population

    # دالة الطفرة
    def mutation(self):
        # يتم حساب الطفرة ونسبتها عن طريق استخدام القناع الخاص في الxor
        mask = np.random.choice([0, 1], (self.pop_size, self.chromosome_length), p=[1 - self.pm, self.pm])
        pop_array = np.array(self.population)
        self.population = np.bitwise_xor(pop_array, mask)

    # فك تشفير الكروموسومات
    def decode(self, chromosome):
        def normal(gene):
            pow2_list = np.array([2 ** i for i in range(0, len(gene))])[::-1]
            decimal = np.sum(pow2_list * gene)
            normalized = decimal / (2 ** len(gene) - 1)
            return normalized

        # فصل الكروموسومات الى جينات
        genes = np.split(chromosome, np.cumsum(self.bit_length)[:-1])

        # نقل القيم الى المستوى المناسب
        normalized_genes = list(map(normal, genes))

        # تحديد المجال للقيم المستخدمة
        decoded_chromosome = np.full((self.dimension), 0, dtype=float)
        for i in range(self.dimension):
            xi = normalized_genes[i]
            xi = self.low[i] + (self.hi[i] - self.low[i]) * xi
            decoded_chromosome[i] = xi

        return decoded_chromosome

    # فك التشفير للجيل
    def decode_population(self):
        self.decoded_population = np.array(
            [self.decode(chromosome) for chromosome in self.population])

    # دالة حساب الاقوى
    def calculate_fitness(self):
        self.population_fitness = np.array(
            [self.fitness_function(chromosome) for chromosome in self.decoded_population])

    # دالة افضل حل الى الان
    def update_best_so_far(self, max_fitness, solution):
        if len(self.best_so_far) == 0:
            self.best_so_far.append(max_fitness)
            self.best_so_far_solution = solution
            return True

        if self.best_so_far[-1] < max_fitness:
            self.best_so_far.append(max_fitness)
            self.best_so_far_solution = solution
        else:
            self.best_so_far.append(self.best_so_far[-1])

    def start(self, plot):
        for iteration in range(self.max_iter):
            # فك تشفير الرموز المستخدمة
            self.decode_population()

            # حساب القيم عن طريق الدالة
            self.calculate_fitness()

            # افضل حل وافضل قيم
            max_fitness = np.max(self.population_fitness)
            index = np.where(self.population_fitness == max_fitness)[0][0]
            optimal_solution = self.decoded_population[index]

            # تحديث الاحصائيات
            self.update_best_so_far(max_fitness, optimal_solution)
            self.average_fitness.append(np.mean(self.population_fitness))

            #  طباعة النتائج عن كل جيل مذكور في الاس
            if plot and iteration in [1, 20, 50]:
                x, y = self.decoded_population.T
                plt.scatter(x, y)

            # طباعة التتبع والقيم خلال وقت التشغيل
            if self.log:
                print('الدورة:', iteration)
                print('اقصى قيمة:', max_fitness, 'الحل:', optimal_solution)
                print('الافضل الى الان:', self.best_so_far[-1], 'عند', list(self.best_so_far_solution))
                print('-' * 70)

            # حساب نسبة الاختيار من الجيل
            selection_probability = self.population_fitness / np.sum(self.population_fitness)

            pool = self.selection(self.pop_size, self.population, selection_probability)

            self.population = self.crossover(pool)

            self.mutation()

        if plot:
            print('العرض..')
            plt.legend(('Gen 1', 'Gen 2', 'Gen 3'))
            plt.show()

        return self.best_so_far, self.average_fitness

#  دالة البقاء للاقوى تم استخدام ال KNN
def func(chromosome):
    # خوارزمية ال س-الجيران الاقرب
    # تحديد المتغيرات المستخدمة في خوارزمية س-الجيران الأقرب
    x1, x2 = chromosome

    # قراءة البيانات
    dataset = pd.read_csv("diabetes.csv")

    # تحديد الصفات والامثله الداخله في الاختبار من البيانات
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    # تقسيم البيانات الى قسم للتدريب وقسم للاختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # تحويل القيم خارج النطاق
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # تحويل القيم من عشرية الى صحيحة
    x1 = int(x1)

    # استدعاء خوارزمية ال س-الجيران الاقرب
    classifier = KNeighborsClassifier(n_neighbors = x1)

    # تدريب النموذج
    classifier.fit(X_train, y_train)

    # تحديد القيمة المثالية
    knn_score = classifier.score(X_test, y_test)
    #print("النتيجة التي توصل لها النموذج", knn_score, " عدد الجيران المستخدمين n_neighbors: ", x1)
    return knn_score

# متغير لافضل قيم
best_so_far = []

# متغير لمتوسط القيم
average = []

# تحديد عدد الدورات لحساب المتوسط الحسابي
runs = 5
for i in range(runs):
    plot = False
    if i == 1:
        plot = True
    # انشاء كائن من الكلاس BGA
    my_ga = BGA()

    # استدعاء دالة البقاء للاقوى
    my_ga.set_fitness(func)
    try:
        bsf, avrg = my_ga.start(plot)
    except ValueError:
        print("تاكد من ابعاد القيم المستخدمة", len(bsf), "avrg", len(avrg))
    print("افضل نتائج في الجيل رقم   ", i , "  للمتغير bsf : ", bsf , "وللمتغير المتوسط:", avrg)

    # تجميع القيم الافضل
    best_so_far.append(bsf)

    # تجميع المتوسطات
    average.append(avrg)

# تحويل النتائج الى مصفوفة
best_so_far = np.array(best_so_far)

# تحويل النتائج الى مصفوفة
average = np.array(average)

# حساب المتوسط الحسابي لافضل قيم
best_so_far = np.mean(best_so_far, axis=0)

# حساب المتوسط الحسابي لكل المتوسطات
average = np.mean(average, axis=0)

# استعراض القيم على شكل
plt.plot(best_so_far)
plt.plot(average)
plt.legend(('Best So far', 'Average'))
plt.show()