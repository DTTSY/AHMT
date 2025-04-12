from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd


def preprocess_data(dataset_path: str, doPerturb: bool, dropna: bool = True, frac=1):
    df = pd.read_csv(dataset_path)
    # shaffle the data
    # np.random.seed(0)
    # df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    # drop the duplicate rows
    if dropna:
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        # 将index重置，drop=True表示删除原来的index列
        df.reset_index(inplace=True, drop=True)
    # drop the rows with missing values
    # df.reset_index(inplace=True, drop=True)

    # 取10%的数据
    df = df.iloc[:int(np.floor(len(df) * frac))]

    data = df.iloc[:, :-1]
    # 将data每列的数据类型转换为float
    data = data.astype(float)
    label = df.iloc[:, -1]

    le = LabelEncoder()
    le = le.fit(label)
    label = le.transform(label)
    # data = MinMaxScaler().fit_transform(data)
    # data = StandardScaler().fit_transform(data)
    k = le.classes_.shape[0]

    if doPerturb:
        random_matrix = np.random.rand(data.shape[0], data.shape[1]) * 1e-7
        data = data + random_matrix
    # data = pd.DataFrame(data)
    _dlabel = pd.DataFrame(label)
    # _data = pd.DataFrame(data)
    _data = pd.concat([data, _dlabel], axis=1)
    # 统计data中的缺失值
    # print(f'{dataset_path.split('/')[-1]}: 缺失值个数：{data.isnull().sum().sum()}')
    return data, label, k, _data


def get_data_from_local(dataset_path: str, doPerturb: bool = False, dropna: bool = True, frac=1):
    return preprocess_data(dataset_path, doPerturb, dropna=True, frac=frac)


def list_available_datasets():
    datasets = list_available_datasets()
    print(datasets)


def load_data(name: str):
    original = fetchData(name)
    X, y, n_classes = process_data(original)
    return X, y, n_classes


def process_data(original: pd.DataFrame):
    # 将X，y的类型是DataFrame，将他们合成一个DataFrame
    print('begain process_data', original.shape)

    # 去除缺失值
    original.dropna(inplace=True)
    #  对X去重，X是pandas.DataFrame
    original.drop_duplicates(inplace=True)
    print('after drop_duplicates', original.shape)
    # 将X和y分开
    X = original.iloc[:, :-1].values
    y = original.iloc[:, -1].values
    # 对X做Max-Min处理
    X = MinMaxScaler().fit_transform(X)
    # 对y做标签编码处理
    y = LabelEncoder().fit_transform(y)
    # y的类别数y是np.array
    yu = np.unique(y)
    n_classes = yu.shape[0]
    return X, y, n_classes


def fetchData(name: str):
    # check which datasets can be imported
    # list_available_datasets()
    # import dataset
    # heart_disease = fetch_ucirepo(id=45)
    heart_disease = fetch_ucirepo(name=name)
    # access data
    print(heart_disease.metadata.abstract)
    print(heart_disease.metadata.additional_info.summary)

    return heart_disease.data.original
    # access metadata
    # print(heart_disease.metadata.uci_id)
    # print(heart_disease.metadata.num_instances)

    # access variable info in tabular format
    # print(heart_disease.variables)


def adjConcat(a, b):
    '''
    将a,b两个矩阵沿对角线方向斜着合并，空余处补零[a,0.0,b]
    得到a和b的维度，先将a和b*a的零矩阵按行（竖着）合并得到c，再将a*b的零矩阵和b按行合并得到d
    将c和d横向合并
    '''
    lena = len(a)
    lenb = len(b)
    # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
    left = np.row_stack((a, np.zeros((lenb, lena))))
    # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
    right = np.row_stack((np.zeros((lena, lenb)), b))
    result = np.hstack((left, right))  # 将左右矩阵水平拼接
    return result


def get_syc_data(blockSize: list, blockNum: int):
    if len(blockSize) < 1:
        return None
    result = np.ones((blockSize[0], blockSize[0]))
    for _ in range(blockNum):
        a = np.ones((blockSize[0], blockSize[0]))
        result = adjConcat(result, a)
    return result, _, blockNum


if __name__ == '__main__':
    raise Exception('This is a module, not a script!')
    data = ['Balance', 'banknote', 'breast', 'dermatology', 'diabetes', 'ecoli', 'glass', 'haberman', 'ionosphere', 'iris',
            'led', 'mfeat karhunen', 'mfeat zernike', 'musk', 'pima', 'seeds', 'segment', 'soybean', 'thyroid', 'vehicle', 'wine']
    for i in data:
        print(i)
        X, y, n_classes = load_data('i')
        # 水平合并
        print(np.hstack((X, y.reshape(-1, 1))).shape)
