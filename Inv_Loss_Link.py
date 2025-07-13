import geopandas as gpd 
from itertools import product
import pickle 
import pandas as pd
from shapely.geometry import MultiPoint
import numpy as np
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os 
import tqdm
from tqdm.notebook import tqdm
import warnings
import requests
import zipfile
import io
import os

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

class Inventory_Loss():
    """
    The class to compute hurricanr los given an inventory and a set of hazard wind speed maps and flood depth estimations.

    :param str cwd_I: Path to the Inventory data.
    :param str cwd_I: Path to the Hazard data.
    :ivar str cwd: Stores the working directory path.
    """
    def __init__(self,cwd='.', Download=False):
        self.cwd=cwd
        # Download the sample file to run the code


        if Download:
            download_url = "https://www.dropbox.com/scl/fi/ig4nsej4chjdkch26f3jz/CHEER_Safe_data.zip?rlkey=exn7uvelo5ar3qpdcau7n3snb&st=f48kc6md&dl=1"
            destination_folder = self.cwd
            os.makedirs(destination_folder, exist_ok=True)
            self.download_and_unzip(download_url, destination_folder)

        # Establish some required file directory
        self.mk_dir(os.path.join(cwd,'Test'))
        self.mk_dir(os.path.join(cwd,'Inv_Updated_Hazard'))
        self.mk_dir(os.path.join(cwd,'CHEER_Safe_data'))
        self.mk_dir(os.path.join(cwd,'Zones'))
        self.mk_dir(os.path.join(cwd,'Hazards'))
        self.mk_dir(os.path.join(cwd,'Intermediate_outputs'))
        self.mk_dir(os.path.join(cwd,'Inventory'))
        self.mk_dir(os.path.join(cwd,'User_Output'))
        self.mk_dir(os.path.join(cwd,'Loss_estimates'))

        
        #cwd_inv=cwd_I
        #cwd_haz=cwd_haz
        #self.cwd_look=os.path.join(self.cwd,'CHEER_Safe data',cwd_look)
        self.mph_to_id = {
            0: 1,
            50: 2,
            55: 3,
            60: 4,
            65: 5,
            70: 6,
            75: 7,
            80: 8,
            85: 9,
            90: 10,
            95: 11,
            100: 12,
            105: 13,
            110: 14,
            115: 15,
            120: 16,
            125: 17,
            130: 18,
            135: 19,
            140: 20,
            155: 21,
            170: 22,
            185: 23,
            200: 24,
            215: 25
        }

        self.flood_depth_to_id = {
            0: 1,    1: 2,    2: 3,    3: 4,    4: 5,    5: 6,    6: 7,    7: 8,    8: 9,
            9: 10,  10: 11,  11: 12,  12: 13,  13: 14,  14: 15,  15: 16,  16: 17,  17: 18,
            18: 19,  19: 20,  20: 21,  21: 22,  22: 23,  23: 24,  24: 25,  25: 26,  26: 27,
            27: 28,  28: 29,  29: 30,  30: 31,  31: 32,  32: 33,  33: 34,  34: 35,  35: 36,
            36: 37,  37: 38,  38: 39,  39: 40,  40: 41,  41: 42,  42: 43,  43: 44,  44: 45,
            45: 46,  46: 47,  47: 48,  48: 49,  49: 50,  50: 51,  51: 52,  52: 53,  53: 54,
            54: 55,  55: 56,  56: 57,  57: 58,  58: 59,  59: 60,  60: 61,  61: 62,  62: 63,
            63: 64,  64: 65,  65: 66,  66: 67,  67: 68,  68: 69,  69: 70,  70: 71,  71: 72,
            72: 73,  73: 74,  74: 75,  75: 76,  76: 77,  77: 78,  78: 79,  79: 80,  80: 81,
            81: 82,  82: 83,  83: 84,  84: 85,  85: 86,  86: 87,  87: 88,  88: 89,  89: 90,
            90: 91,  91: 92,  92: 93,  93: 94,  94: 95,  95: 96,  96: 97,  97: 98,  98: 99,
            99: 100, 100: 101, 101: 102, 102: 103, 103: 104, 104: 105, 105: 106, 106: 107,
        107: 108, 108: 109, 109: 110, 110: 111, 111: 112, 112: 113, 113: 114, 114: 115,
        115: 116, 116: 117, 117: 118, 118: 119, 119: 120, 120: 121, 130:122,140: 123, 150: 124,
        160: 125, 170: 126, 180: 127, 190: 128, 200: 129, 210: 130, 220: 131, 230: 132,
        240: 133, 250: 134, 260: 135, 270: 136
        }

        self.damage_lavels = np.array([
            0.000, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036,
            0.040, 0.060, 0.080, 0.100, 0.120, 0.140, 0.160, 0.180, 0.200, 0.220,
            0.240, 0.280, 0.320, 0.360, 0.400, 0.440, 0.480, 0.520, 0.560, 0.600,
            0.640, 0.680, 0.720, 0.760, 0.800, 0.840, 0.880, 0.920, 0.960, 1.000,
            1.040, 1.080, 1.120, 1.160, 1.200, 1.240, 1.280, 1.320, 1.360, 1.400
        ])


    def download_and_unzip(self,url, extract_to):
        # Convert Dropbox sharing link to direct download link
        if 'dropbox.com' in url and 'dl=0' in url:
            url = url.replace('dl=0', 'dl=1')

        # Download the file
        print("Downloading sample files...")
        response = requests.get(url)
        response.raise_for_status()

        # Unzip the file while preserving folder structure
        print("Unzipping...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_to)

        print(f"Extracted to: {extract_to}")


    def mk_dir(self,cwd):

        """
        Check if a directory exists, if not, makes it.

        :param str cwd: the directory address

        """
        
        try:
            os.mkdir(cwd)
        except:
            pass
    def interpolate_mph_to_id(self,x, table):
        keys = sorted(table.keys())
        
        if x <= keys[0]:
            return table[keys[0]]
        if x >= keys[-1]:
            return table[keys[-1]]
        
        for i in range(len(keys) - 1):
            x0, x1 = keys[i], keys[i + 1]
            if x0 <= x <= x1:
                y0, y1 = table[x0], table[x1]
                # Linear interpolation of ID
                return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            
    def Match_with_four(self,A,B):
        A.geometry=A.geometry.centroid
        # Step 1: Get coordinate arrays
        A_coords = np.array([A.geometry.x, A.geometry.y]).T
        B_coords = np.array([B.geometry.x, B.geometry.y]).T
        # Step 2: Build KDTree and query nearest neighbors
        tree = cKDTree(B_coords)
        dists, idxs = tree.query(A_coords, k=4)
        # Step 3: Flatten and expand indices
        A_idx_repeated = np.repeat(A.index.values, 4)
        B_idx_flat = idxs.reshape(-1)
        # Step 4: Get the selected B rows
        B_selected = B.iloc[B_idx_flat].copy()
        B_selected['A_index'] = A_idx_repeated
        B_selected['rank'] = np.tile(np.arange(1, 5), len(A))
        B_selected['distance'] = dists.reshape(-1)
        # Step 5: Merge A attributes based on A_index
        A_subset = A[['ID_I']].copy()
        A_subset['A_index'] = A.index
        # Merge attributes from A into the result
        expanded = B_selected.merge(A_subset, on='A_index', how='left')
        expanded = expanded.reset_index(drop=True)[['ID_I','ID_H','distance']]
        return expanded 


    def Define_new_table(self,cwd_look):
        # Open the condifuration .csv file (see documentation)
        Inv=pd.read_csv(os.path.join(self.cwd,'CHEER_Safe_data','All_configs.csv'))

        # Many following details are based on: Peng, Jiazhen. Modeling natural disaster risk management: Integrating the roles of insurance and retrofit and multiple stakeholder perspectives. University of Delaware, 2013.
        # We refer to that as Jiazhen et al. in the following 


        # "Archtype" configurations (Jiazhen et al. Table 3.1): 
        cond1 = (Inv['Story'] == 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond2 = (Inv['Story'] == 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)
        cond3 = (Inv['Story'] == 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond4 = (Inv['Story'] == 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)
        cond5 = (Inv['Story'] != 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond6 = (Inv['Story'] != 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)
        cond7 = (Inv['Story'] != 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond8 = (Inv['Story'] != 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)

        # Choices corresponding to each condition (Jiazhen et al. Table 3.1): 
        choices = [1, 3, 2, 4, 5, 7, 6, 8]

        # Apply vectorized logic
        import numpy as np
        Inv["M"] = np.select([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8], choices, default=np.nan)
        
        ### C tables: (Jiazhen et al. Table 3.6 and 3.7): 
        rulesN = {
            "pre-1968": {
                "Roof covering": {1: 1, 2: 0},
                "Roof sheathing": {1: 1, 2: 0},
                "Roof-to-wall": {1: 1, 2: 0},
                "Openings": {1: 1, 2: 0, 3: 0},
                "Walls": {1: 1, 2: 0},
                "Flood": {1: 1, 2: 0}
            },
            "1968-1997": {
                "Roof covering": {1: 0.1, 2: 0.9},
                "Roof sheathing": {1: 1, 2: 0},
                "Roof-to-wall": {1: 0.5, 2: 0.5},
                "Openings": {1: 0.8, 2: 0.1, 3: 0.1},
                "Walls": {1: 0.7, 2: 0.3},
                "Flood": {1: 0.2, 2: 0.2, 3: 0.2,4:0.4}
            },
            "1997-2010": {
                "Roof covering": {1: 0, 2: 1},
                "Roof sheathing": {1: 0.5, 2: 0.5},
                "Roof-to-wall": {1: 0.1, 2: 0.9},
                "Openings": {1: 0.5, 2: 0.25, 3: 0.25},
                "Walls": {1: 0.1, 2: 0.9},
                "Flood": {1: 0, 2: 0.1, 3: 0.1, 4: 0.8}
            }
        }

        rulesF = {
            "pre-1968": {
                "Roof covering": {1: 1, 2: 0},
                "Roof sheathing": {1: 1, 2: 0},
                "Roof-to-wall": {1: 1, 2: 0},
                "Openings": {1: 1, 2: 0, 3: 0},
                "Walls": {1: 1, 2: 0},
                "Flood": {1: 1, 2: 0}
            },
            "1968-1997": {
                "Roof covering": {1: 0.9, 2: 0.1},
                "Roof sheathing": {1: 1, 2: 0},
                "Roof-to-wall": {1: 0.9, 2: 0.1},
                "Openings": {1: 0.9, 2: 0.05, 3: 0.05},
                "Walls": {1: 1.0, 2: 0},
                "Flood": {1: 0.85, 2: 0.05, 3: 0.05,4:0.05}
            },
            "1997-2010": {
                "Roof covering": {1: 0.8, 2: 0.2},
                "Roof sheathing": {1: 0.9, 2: 0.1},
                "Roof-to-wall": {1: 0.8, 2: 0.2},
                "Openings": {1: 0.8, 2: 0.1, 3: 0.1},
                "Walls": {1: 0.9, 2: 0.1},
                "Flood": {1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05}
            }
        }


        # Map year bins to categories ((Jiazhen et al. Table 3.6 and 3.7):
        def classify_year(med_yr_blt):
            year = int(med_yr_blt)
            if year < 1968:
                return 'pre-1968'
            elif year < 1997:
                return '1968-1997'
            else:
                return '1997-2010'



        # Building up "C" combinations (Jiazhen et al. Table A.1): 
        components = ["Roof covering", "Roof sheathing", "Roof-to-wall","Openings" ,"Walls","Flood"]
        configs = [1, 2, 3, 4]

        # Function to establish Jiazhen et al. Table A.1:
        def get_config_matrix(row):
            yr_category = classify_year(row.Year)
            rule_set = rulesN if row['D'] < 2 else rulesF
            C = [[rule_set[yr_category][c].get(co, 0) for co in configs] for c in components]
            return C

        C_values = Inv.apply(get_config_matrix, axis=1).to_numpy()
        C_values = np.array(C_values.tolist())

        # Permutation in Table 3.6 and 3.7)
        configs=[1,2,3,4]
        configs2=[[1,2],[1,2],[1,2],[1,2,3],[1,2],[1,2,3,4]]
        combs = np.array(list(product(configs2[5], configs2[0], configs2[1], configs2[2], configs2[3], configs2[4])))-1

        part1 = C_values[:, 5, combs[:, 0]]
        part2 = C_values[:, 0, combs[:, 1]]
        part3 = C_values[:, 1, combs[:, 2]]
        part4 = C_values[:, 2, combs[:, 3]]
        part5 = C_values[:, 3, combs[:, 4]]
        part6 = C_values[:, 4, combs[:, 5]]

        # This return the probability of each C value for a given M value 
        K1_array = part1 * part2 * part3 * part4 * part5 * part6  
        K1 = K1_array.tolist()

        # Check if the sum adds up yo 1
        final_sum = K1_array[-1].sum()
        print("Probabilities summation (should be one; small numerical errors are fine):",final_sum)

        # Define the "Z" values (they can be simply put from 1 to N, the size of configurations)
        Inv['Z']=np.arange(0,len(Inv),1)+1
        Inv.index=np.arange(0,len(Inv),1)

        #Update the configuration .csv file by "Z" values and store it
        Inv.to_parquet(os.path.join(self.cwd,'Test','Inv_Test.parquet'))


        # Building new lookup_table based on "Z"
        
        # Load CHEER-Safe lookup table
        with open(os.path.join(self.cwd,'CHEER_Safe_data',cwd_look), 'rb') as file:
            data = pickle.load(file)

        
        transformed_lookup = {}
        D=0
        zet=1
        for enu_1 in Inv.index:
            #Global damage ratio: dot product of all 25 damage states by their probability
            gd_r=0
            m=Inv.iloc[enu_1].M
            for c in range(1,193):
                gd_r+=np.einsum('ijk,k->ij', data[m,c], self.damage_lavels)*K1_array[enu_1,c-1]

            transformed_lookup[zet] = gd_r
            zet+=1
        
        # Save the new lookup table that maps: Dict[Z][Wind_id, Flood_id]->Global damage ratio (i.e., summation of all 25 damage states by their probability)
        with open(os.path.join(self.cwd,'Intermediate_outputs','lookup_N.pkl'), 'wb') as f:
            pickle.dump(transformed_lookup,f)


        Inv['D_bin'] = Inv['D'].apply(lambda d: 0 if d < 2 else 1)
        Inv['Year_bin'] = Inv['Year'].apply(lambda d: 0 if d < 1968 else (1 if d < 1997 else 2))

        z_transform = {}

        zet=0
        for i in range(len(Inv)):
            z_transform[(Inv.iloc[i].D_bin,Inv.iloc[i].Year_bin,Inv.iloc[i].M)] = i+1
            zet+=1
        with open(os.path.join(self.cwd,'Intermediate_outputs','Z_transform1.pkl'), 'wb') as f:
            pickle.dump(z_transform,f)

        print('Dictionary to map inventory to configurations is created at:\t'+os.path.join(self.cwd,'Intermediate_outputs','lookup_N.pkl'))
        print('New look-up table (configuration (Z) -> global loss ratio) is created at:\t'+os.path.join(self.cwd,'Intermediate_outputs','Z_transform1.pkl'))

    def Inv_Z_Transformation(self,cwd_inv,blgtype_col=['bldgtype',['W']],roof_col=['roof_shape_1',['Cross Gable', 'Complex Gable','Gable'],['Hip','Cross Hip']],story_col=['Story_AI',[1],[2]],garage_col=['Garage',[0],[1]],res_col=['PRIM_OCC',['Single Family Dwelling']],year_col=['med_yr_blt'],Coast_D_col=['Coast_D (miles)',[1]],val_struct_col=['val_struct'],Mode='Compute'):
        """
        Method to transform building inventory to archetypes, defined by C and M

        :return: None
        """
        if Mode=='Test':
            Inv=pd.read_parquet(os.path.join(self.cwd,'Test','Inv_Test.parquet'))
            blgtype_col=['Type',['W']]
            roof_col=['Roof',['Gable'],['Hip']]
            story_col=['Story',[1],[2]]
            garage_col=['Garage',[0],[1]]
            res_col=['Occ',['Single']]
            year_col=['Year']
            Coast_D_col=['D',[1]]
            val_struct_col=['val_struct']
        else:
            try:
                try:
                    Inv=gpd.read_parquet(os.path.join(self.cwd,'Inventory',cwd_inv+'.parquet'))
                except:
                    Inv=gpd.read_file(os.path.join(self.cwd,'Inventory',cwd_inv+'.shp'))
            except:
                print('Please make sure your inventory is either in .shp ro .(g)parquet format')
        
        # Building type
        bldg_dict={key: 'W' for key in blgtype_col[1]}
        Inv[blgtype_col[0]]=Inv[blgtype_col[0]].map(bldg_dict)
        # Roof type
        roof_dict={key: 'Gable' for key in roof_col[1]} | {key: 'Hip' for key in roof_col[2]}
        Inv[roof_col[0]]=Inv[roof_col[0]].map(roof_dict)
        Inv=Inv.rename(columns={roof_col[0]:'Roof'})
        # Story
        story_dict={key: 1 for key in story_col[1]} | {key: 2 for key in story_col[2]}
        Inv[story_col[0]]=Inv[story_col[0]].map(story_dict)
        Inv=Inv.rename(columns={story_col[0]:'Story'})
        # Garage
        garage_dict={key: 0 for key in garage_col[1]} | {key: 1 for key in garage_col[2]}
        Inv[garage_col[0]]=Inv[garage_col[0]].map(garage_dict)
        Inv=Inv.rename(columns={garage_col[0]:'Garage'})
        # Residency
        res_dict={key: 'Single' for key in res_col[1]}
        Inv[res_col[0]]=Inv[res_col[0]].map(res_dict)

        # ys uilt
        Inv=Inv.rename(columns={year_col[0]:'Year'})

        # Coastal D(mile)
        Inv=Inv.rename(columns={Coast_D_col[0]:'Coast_D'})
        Inv['Coast_D']=Inv['Coast_D']*Coast_D_col[1]

        # value

        Inv=Inv.rename(columns={val_struct_col[0]:'val_struct_col'})

        # Filter the inventory
        Inv=Inv.dropna()
        if len(Inv)==0:
            print('Please dounle check the above details, as no single row matches that')

        # Identify M to further transform Z
        # "Archtype" configurations (Jiazhen et al. Table 3.1): 
        cond1 = (Inv['Story'] == 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond2 = (Inv['Story'] == 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)
        cond3 = (Inv['Story'] == 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond4 = (Inv['Story'] == 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)
        cond5 = (Inv['Story'] != 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond6 = (Inv['Story'] != 1) & (Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)
        cond7 = (Inv['Story'] != 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] == 1)
        cond8 = (Inv['Story'] != 1) & (~Inv['Roof'].str.contains("Hip", na=False)) & (Inv['Garage'] != 1)

        # Choices corresponding to each condition (Jiazhen et al. Table 3.1): 
        choices = [1, 3, 2, 4, 5, 7, 6, 8]

        # Apply vectorized logic
        import numpy as np
        Inv["M"] = np.select([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8], choices, default=np.nan)

        # Add Z to each row
        Inv['Coast_D_bin'] = Inv['Coast_D'].apply(lambda d: 0 if d < 2 else 1)
        Inv['Year_bin'] = Inv['Year'].apply(lambda d: 0 if d < 1968 else (1 if d < 1997 else 2))

        with open(os.path.join(self.cwd,'Intermediate_outputs','Z_transform1.pkl'), 'rb') as f:
                z_transform=pickle.load(f)

        Inv['Z']=Inv[['M','Year_bin','Coast_D_bin']].apply(lambda x: z_transform[x['Coast_D_bin'],x['Year_bin'],x['M']],axis=1)

        if Mode=='Test':
            Z=Inv[['M','Year_bin','Coast_D_bin']].apply(lambda x: z_transform[x['Coast_D_bin'],x['Year_bin'],x['M']],axis=1)
            if np.array_equal(Inv['Z'],Z):
                    print('Z transformation \t','Pass')
            else:
                    print('Z transformation: \t','Fail')
        else:
            # Store transformed Inventory for further analysis
            Inv.to_parquet(os.path.join(self.cwd,'Intermediate_outputs',cwd_inv.replace('.parquet','')+'_Z.parquet'))
            print('Updated inventory (filtered for archetypes and updated with Z values) stored at:\t',os.path.join(self.cwd,'Intermediate_outputs',cwd_inv.replace('.parquet','')+'_Z.parquet'))
            
    def Hazard_Inv_map(self,cwd_inv,cwd_haz,W_ind=3,I_ind=4,Tr_F=39.3701,Tr_W=2.23694,Average_Mode='Inverse_d4',Mode=''):
        # Make directories to store the updated inventory with the flood and wind data
        self.mk_dir(os.path.join(self.cwd,'Inv_Updated_Hazard'))
        self.mk_dir(os.path.join(self.cwd,'Inv_Updated_Hazard',cwd_haz))

        if Mode=='Test':
            # Read the Inventory and Hazard data
            Inv=gpd.read_parquet(os.path.join(self.cwd,'Test','Hazard_match_sample.parquet'))
            Haz = gpd.read_parquet(os.path.join(self.cwd,'Test','Test.parquet'))
        else:
            Inv=gpd.read_parquet(os.path.join(self.cwd,'Intermediate_outputs',cwd_inv+'_Z.parquet'))
            #read one hazard data to build the matching pattern
            H_add=os.listdir(os.path.join(self.cwd,'Hazards',cwd_haz))
            try:
                Haz = gpd.read_parquet(os.path.join(self.cwd,'Hazards',cwd_haz,H_add[0]))
            except:
                Haz= gpd.read_file(os.path.join(self.cwd,'Hazards',cwd_haz,H_add[0]))

        # Create a rectangular polygon (bounding box)
        from shapely.geometry import box
        boundary = box(*Haz.total_bounds)
        Inv=Inv[Inv.geometry.centroid.within(boundary)]
        # Store geometry and value of structure to save sapce
        Inv.index=np.arange(0,len(Inv),1)
        Inv[['geometry','val_struct_col']].to_parquet(os.path.join(self.cwd+'Intermediate_outputs',cwd_haz+'_'+'Inv_'+cwd_inv+'.parquet'))
        # Instantiate index and an identifier column
        Inv['ID_I']=np.arange(0,len(Inv),1)
        Inv.index=np.arange(0,len(Inv),1)
        Haz['ID_H']=np.arange(0,len(Haz),1)
        Haz.index=np.arange(0,len(Haz),1)


        # find the wind and flood column names inisde the hazard data
        W_col=Haz.columns[W_ind]
        I_col=Haz.columns[I_ind]
        #rename the hazard column for simplicity
        Haz=Haz.rename(columns={W_col:'Wind',I_col:'Flood'})

        #make a more manageable hazard dataframe
        haz=Haz[['geometry','Wind','Flood','ID_H']]

        #make a more manageable inventory dataframe
        inv=Inv[['geometry','ID_I','Z','val_struct_col']]
        pas=0
        # Define the match pattern 
        if Average_Mode=='Inverse_d4':
            match_pattern=self.Match_with_four(inv,haz)
            matched=inv[['ID_I','Z']].merge(match_pattern.merge(haz,on='ID_H').drop(columns=['geometry']),on='ID_I')
            pas=1
        else:
            print('The chosen average system does not exist; only "Inverse_d4" is available at the moment')
            pas=0


        
        if Mode=='Test' and pas==1:
            # Averaging 
            num_col1=['Wind','Flood']
            matched['distance_inv']=1/matched['distance']
            matched[num_col1] = matched[num_col1].transform(lambda x: x * matched['distance_inv'])
            # Sum weights
            group_sum = matched.groupby('ID_I', as_index=False).agg(b_sum=('distance_inv', 'sum'))
            # Sm Wind and Flood attributes
            EE=matched.groupby('ID_I', as_index=False)[num_col1].sum()
            EE['sum']=group_sum['b_sum']
            #Average
            EE[num_col1] = EE[num_col1].transform(lambda x: x / EE['sum'])
            # Merge
            Inv=Inv.merge(EE[['Wind','Flood','ID_I']])
            if np.allclose(Inv['Wind'], Inv['W'], atol=0.0001) and np.allclose(Inv['Flood'], Inv['I'], atol=0.0001):
                print('Test for hazard matching is:\t: Passed')
            else:
                print('Test for hazard matching is:\t: Failed')    

        elif pas==1:
            for h_add in tqdm(H_add,desc='Hazard instances:'):
                try:
                    Haz = gpd.read_parquet(os.path.join(self.cwd,'Hazards',cwd_haz,h_add))
                except:
                    Haz = gpd.read_file(os.path.join(self.cwd,'Hazards',cwd_haz,h_add))
                Haz['ID_H']=np.arange(0,len(Haz),1)
                Haz.index=np.arange(0,len(Haz),1)
                Haz=Haz.rename(columns={W_col:'Wind',I_col:'Flood'})
                haz=Haz[['geometry','Wind','Flood','ID_H']]
                haz[haz==-9999]=0
                # Instantiate index and an identifier column
                Haz['ID_H']=np.arange(0,len(Haz),1)
                Haz.index=np.arange(0,len(Haz),1)

                # Define the match pattern 
                matched=inv[['ID_I','Z']].merge(match_pattern.merge(haz,on='ID_H').drop(columns=['geometry']),on='ID_I')

                num_col1=['Wind','Flood']
                matched['distance_inv']=1/matched['distance']
                matched[num_col1] = matched[num_col1].transform(lambda x: x * matched['distance_inv'])   
                # Sum weights
                group_sum = matched.groupby('ID_I', as_index=False).agg(b_sum=('distance_inv', 'sum'))
                # Sm Wind and Flood attributes
                EE=matched.groupby('ID_I', as_index=False)[num_col1].sum()
                EE['sum']=group_sum['b_sum']
                #Average
                EE[num_col1] = EE[num_col1].transform(lambda x: x / EE['sum'])
                # Merge
                inv_C=inv.merge(EE[['Wind','Flood','ID_I']])[['Wind','Flood','Z','val_struct_col']]
                #print(np.mean(inv_C['Wind']))
                #Transformed Wind and Flood depth to ID's 25/136 (no actual documentation, please look at CHEER-Safe details)
                inv_C['Flood_Tr']=inv_C['Flood'].apply(lambda x:self.interpolate_mph_to_id(x*Tr_F, self.flood_depth_to_id))
                inv_C['Wind_Tr']=inv_C['Wind'].apply(lambda x:self.interpolate_mph_to_id(x*Tr_W, self.mph_to_id))
                inv_C[['Flood_Tr','Wind_Tr','Z','val_struct_col']].to_parquet(os.path.join(self.cwd,'Inv_Updated_Hazard',cwd_haz,'Inv_'+h_add))
        if pas==0:
            print('Averaging system does not exist, and thus no result is saved')

    def Loss_estimate(self,cwd_haz,cwd_inv,zone=''):
        Inv_dir=os.path.join(self.cwd,'Inv_Updated_Hazard',cwd_haz)
        L=os.listdir(Inv_dir)

        # Open the 
        with open(os.path.join(self.cwd,'Intermediate_outputs','lookup_N.pkl'), 'rb') as f:
            dic=pickle.load(f)

        #A list to store the Inventory damage summation in B$
        DT=[]
        #A list to keep the track of hazard incident names
        Name=[]

        # Open the sample Inventory to read values
        Inv_dir=os.path.join(self.cwd,'Inv_Updated_Hazard',cwd_haz)
        Inv=gpd.read_parquet(os.path.join(self.cwd+'Intermediate_outputs',cwd_haz+'_'+'Inv_'+cwd_inv+'.parquet'))

        # Build the required directories
        self.mk_dir(os.path.join(self.cwd,'Loss_estimates'))
        self.mk_dir(os.path.join(self.cwd,'Loss_estimates',cwd_haz))
        self.mk_dir(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory'))
        self.mk_dir(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv.replace('.parquet','')))

        # Build the for user-friendly .csv files
        self.mk_dir(os.path.join(self.cwd,'User_Output'))
        self.mk_dir(os.path.join(self.cwd,'User_Output',cwd_haz))
        self.mk_dir(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory'))
        self.mk_dir(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory',cwd_inv.replace('.parquet','')))

        # If zone level is requested
        if len(zone)>0:
            try:
                try:
                    zones=gpd.read_parquet(os.path.join(self.cwd,'zones',zone+'.parquet'))
                    zone_computation=True
                except:
                    zones=gpd.read_file(os.path.join(self.cwd,'zones',zone+'.shp'))
                    zone_computation=True
            except:
                print('The chosen zone does not exist')
                zone_computation=False
        if zone_computation:
            # Make directory to save
            self.mk_dir(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),'Zones'))
            self.mk_dir(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),'Zones',zone))
            # For user firendly files
            self.mk_dir(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),'Zones'))
            self.mk_dir(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),'Zones',zone))

        for l in tqdm(L,desc='Hazard instances:'):
            # Read the Wind and Flood depth, for the iventory, given a hazard scednario
            W_F=pd.read_parquet(os.path.join(Inv_dir,l))#
            #print(np.mean(W_F['Wind_Tr']))
            W_F[['geometry','val_struct_col']]=Inv[['geometry','val_struct_col']]
            W_F = gpd.GeoDataFrame(W_F, geometry='geometry')
            W_F.set_crs(Inv.crs, inplace=True)
            # Employ the Z-based lookup table to read global damage state (01-140% damage).
            W_F['Damage_ratio']=W_F[['Wind_Tr','Flood_Tr','Z']].apply(lambda x: dic[x['Z']][int(x['Wind_Tr']-1),int(x['Flood_Tr']-1)],axis=1)
            W_F['Loss(B$)']=W_F['Damage_ratio']*W_F['val_struct_col']
            Name.append((l.replace('.parquet','')).replace('Inv_',''))
            DT.append(np.sum(W_F['Loss(B$)']/1000000000))

            W_F[['Damage_ratio','Loss(B$)']].to_parquet(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),l))

            if zone_computation:
                zones.to_crs("EPSG:4326", inplace=True)
                zones = zones.reset_index(drop=True)
                zones['zone_id'] = zones.index

                # Spatial join: assign each point in Inv to a zone
                W_F_centroids = gpd.GeoDataFrame(W_F, geometry='geometry',crs=Inv.crs)
                W_F_centroids['geometry'] = Inv.geometry.centroid
                joined = gpd.sjoin(W_F_centroids, zones[['zone_id', 'geometry']], predicate='within', how='left')

                # Group by zone_id to compute sum of loss and count
                agg = joined.groupby('zone_id').agg(
                    Z_sum=('Loss(B$)', 'sum'),
                    N_count=('Loss(B$)', 'count')
                ).reindex(zones['zone_id'], fill_value=0)

                # Extract results
                zones['Loss(B$)'] = agg['Z_sum'].values
                zones['N'] = agg['N_count'].values
                
                zones.to_parquet(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv,'Zones',zone,l))
                zones[['zone_id','Loss(B$)']].to_csv(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),'Zones',zone,l.replace('.parquet','.csv')))
                
        Loss_df=pd.DataFrame({'Hazard':Name,'Loss (B$)':DT})
        Loss_df.to_parquet(os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv,'Loss_DF.parquet'))

        #Also save it as .csv, for users
        Info=pd.DataFrame({'Hazard':['Total Buildings:'],'Loss (B$)':len(W_F)})
        pd.concat((Loss_df,Info)).to_csv(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory',cwd_inv,'Loss_DF.csv'))

        #Sort zones files, if any for users as csv
        if zone_computation:
            zones[['zone_id','N']].to_csv(os.path.join(self.cwd,'User_Output',cwd_haz,'Inventory',cwd_inv.replace('.parquet',''),'Zones',zone,'zones_id_N.csv'))
            print('Inventory-level are stored at:\t',os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv.replace('.parquet','')))
        if zone_computation:
            print('zone-level are stored at:\t',os.path.join(self.cwd,'Loss_estimates',cwd_haz,'Inventory',cwd_inv,'Zones',zone.replace('.shp','')))