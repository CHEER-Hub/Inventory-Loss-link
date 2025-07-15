# Inventory-Loss Link Module

The **Inventory-Loss Link** module is designed to integrate building inventories and hazard maps with the [CHEER-Safe Loss Estimation Model](https://github.com/CHEER-Hub/LossModel).

Before getting started, please read the documentation:  
📄 [Inventory_Loss Link Documentations](https://cheer-hub.github.io/Inventory-Loss-link/Data_Fusion.html)  
This will help you understand the module and the required data structure to run the model on your dataset.

---

## 🔧 Test Run Instructions

1. Download and unzip the repository.
2. Open and run the `Run_Loss.ipynb` notebook.

---

Once you're familiar with the workflow, you can adapt the code to run other scenarios and datasets using the same process.

---

## 📦 Dependencies

Make sure the following Python packages are installed:

- `pandas`
- `numpy`
- `geopandas`
- `matplotlib`
- `shapely`
- `fiona`
- `pyproj`
- `jupyter`
- `scipy`
- -`requests`
- `tqdm`


You can install them via:

```bash
pip install -r requirements.txt
