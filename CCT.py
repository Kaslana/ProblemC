import numpy as np
import pandas as pd
from colour import MSDS_CMFS, SpectralShape

# 1. 读取数据
df = pd.read_csv("Problem 1.csv")
df.columns = ["wavelength", "intensity"]
df["wavelength"] = df["wavelength"].str.extract(r"(\d+)").astype(int)

# 2. 获取 CIE 1931 色度匹配函数（2°观察者）
cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
cmfs = cmfs.copy().align(SpectralShape(380, 780, 1))  # 对齐波长范围

# 3. 插值匹配函数到数据波长
x_bar = np.interp(df["wavelength"], cmfs.wavelengths, cmfs.values[:, 0])
y_bar = np.interp(df["wavelength"], cmfs.wavelengths, cmfs.values[:, 1])
z_bar = np.interp(df["wavelength"], cmfs.wavelengths, cmfs.values[:, 2])

# 4. 计算三刺激值
S = df["intensity"].values
X = np.sum(S * x_bar)
Y = np.sum(S * y_bar)
Z = np.sum(S * z_bar)

# 5. 计算色品坐标
x = X / (X + Y + Z)
y = Y / (X + Y + Z)

# 6. 使用 McCamy 公式计算 CCT
n = (x - 0.3320) / (y - 0.1858)
CCT = -437 * n ** 3 + 3601 * n ** 2 - 6861 * n + 5514.31

# 7. 输出结果并保存
result = pd.DataFrame([{
    "x": x,
    "y": y,
    "CCT (K)": CCT
}])
result.to_csv("CCT_Result.csv", index=False)

print(result)
