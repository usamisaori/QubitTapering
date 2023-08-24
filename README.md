# QubitTapering

<img src="https://pennylane.ai/_images/qubit_tapering.png">

- tapering_using_sectors.ipynb:
  - qubit tapering 的详细步骤实现，
  - 实现理论依据：https://arxiv.org/pdf/1701.08213.pdf，
  - 同时参考pennylane的实现：https://pennylane.ai/qml/demos/tutorial_qubit_tapering/ ，使用optimal sector
- mytapering.py
  - 对应tapering_using_sectors.ipynb中实现，导出的py模块，
  - 使用：
    - 1. 导入：`from mytapering import *`（确保在mytapering.py同一目录下），
      2. 构建哈密顿量：eg. 构建目标：1.0 XX + 1.5 XZ + 2.0 YZ，代码：`Hami = PauliWords(["XX", "XZ", "YZ"], [1.0, 1.5, 2.0])`
      3. qubit tapering：`Hami_tapered = tapering(Hami, n_electrons)`
      4. 相似的步骤拆分见tapering_using_sectors.ipynb
- mytapering_examples.ipynb:
  - 对比使用pennylane和mytapering的qubit tapering效果
  - 分别考虑四个分子例子：H2, LiH, H2O, BeH2
