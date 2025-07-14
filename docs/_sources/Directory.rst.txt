Directory Structure
===================

The `Inventory` directory is the root of the data system. It organizes files into logically separated subdirectories for efficient access and processing throughout the pipeline. The structure is as follows:

::

    Inventory/
    ├── Raw data/   [R1]
    │   ├── State/  [R2]     
    │      ├── FIPS/    [R3] 
    │         ├── fips_building_microsoft.prquet    [R4]
    │         ├── fips_building_fema.prquet [R5]
    │         ├── fips_building_nsi.prquet  [R6]
    │      ├── Microspft.prquet [R7]
    │      ├── Fema.prquet  [R8]
    │      ├── NSI.prquet   [R9] 
    ├── Transitory data/    [T1]
    │   ├── State/
    │      ├── FIPS/
    │         ├── fips_F_M.prquet   [T2]
    │         ├── fips_F_M_N.prquet [T3]
    │         ├── Fusion_stat.txt   [T4]

Description
-----------

- R1. **Raw_data/**: Directory Containing pre-processed datasources, herein FEMA, Microsoft and NSI.
- R2. **State/**: Data are stored in respective folders labeled with their abbreviate names, e.g., Delaware: DE.
- R3. **FIPS/**: Each state folder is divided into its respective county FIPS code.
- R4. **fips_building_microsoft.prquet**: Pre-processed Microsoft data for each FIPS code.
- R5. **fips_building_fema.prquet**: Pre-processed FEMA data for each FIPS code.
- R6. **fips_building_nsi.prquet**: Pre-processed NSI data for each FIPS code.
- R7. **Microspft.prquet**: The merged FEMA data for the whole State.
- R8. **FEMA.prquet**: The merged Microsoft data for the whole State.
- R9. **NSI.prquet**: The merged Microsoft data for the whole State.
- T1. **Transitory_data/**: Directory Containing processed data obtained from two or more of the datasources.
- T2. **fips_F_M.prquet**: Microsoft-FEMA fused data for each FIPS code.
- T3. **fips_F_M_N.prquet**: Microsoft-FEMA-NSI fused data for each FIPS code.
- T4. **Fusion_stat.txt**: Data fusion process statistics.

Note: The `Inventory` directory is typically assumed to be the current working directory (cwd), assgned by the user as parameter "cwd" in the code. 
