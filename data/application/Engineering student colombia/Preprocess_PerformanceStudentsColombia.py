import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rulelist.rulelist import RuleList
dirctory = 'data/application/Engineering student colombia/data_academic_performance_origianl.xlsx'
data = pd.read_excel(dirctory,na_values='0')
#def ENG(x):
#    return (x['ENG_PRO'] - x['ENG_S11'])/ x['ENG_S11']*100
#def CR(x):
#    return (x['CR_PRO'] - x['CR_S11'])/ x['CR_S11']*100

#data['rel_diff_ENG'] = data.apply(ENG, axis=1)
#data['rel_diff_CR'] = data.apply(CR, axis=1)

variables2stay = ['GENDER', 'EDU_FATHER', 'EDU_MOTHER', 'OCC_FATHER',
       'OCC_MOTHER', 'STRATUM', 'SISBEN', 'PEOPLE_HOUSE',
       'INTERNET', 'TV', 'COMPUTER', 'WASHING_MCH', 'MIC_OVEN', 'CAR', 'DVD',
       'FRESH', 'PHONE', 'MOBILE', 'REVENUE', 'JOB',
       'SCHOOL_NAT', 'SCHOOL_TYPE', 'QR_PRO' ,'ENG_PRO']
data = data[variables2stay]
#df = data.loc[:,['MAT_S11', 'CR_S11', 'CC_S11', 'BIO_S11',
#       'ENG_S11','QR_PRO','CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO', 'FEP_PRO']]
#pd.set_option('display.max_columns', None)
#df.corr()

data.EDU_FATHER = data.EDU_FATHER.apply(lambda x : x if x != 'Not sure' else np.nan)
data.EDU_MOTHER = data.EDU_MOTHER.apply(lambda x : x if x != 'Not sure' else np.nan)
data.OCC_FATHER = data.OCC_FATHER.apply(lambda x : x if x != 'Not sure' else np.nan)
data.OCC_MOTHER = data.OCC_MOTHER.apply(lambda x : x if x != 'Not sure' else np.nan)
data.OCC_MOTHER = data.OCC_MOTHER.apply(lambda x : x if x != 'Not sure' else np.nan)

def transform_stratum(v):
    if v == 'Stratum 1': return 1
    elif v == 'Stratum 2': return 2
    elif v == 'Stratum 3': return 3
    elif v == 'Stratum 4': return 4
    elif v == 'Stratum 5': return 5
    elif v == 'Stratum 6': return 6
    elif np.isnan(v): return np.nan

def transform_education(v):
    if v == 'Ninguno': return 0
    elif v == 'Incomplete primary ': return 0.5
    elif v == 'Complete primary ': return 1
    elif v == 'Incomplete Secundary': return 1.5
    elif v == 'Complete Secundary': return 2.0 # high school
    elif v == 'Incomplete technical or technological': return 2.5 # incomplete Undergrade
    elif v == 'Incomplete Professional Education': return 2.5
    elif v == 'Complete technique or technology': return 3.0
    elif v == 'Complete professional education': return 3.0
    elif v == 'Postgraduate education': return 4
    elif np.isnan(v): return np.nan

def transform_SISBEN(v):
    if v == 'It is not classified by the SISBEN': return 'None'
    elif v == 'Esta clasificada en otro Level del SISBEN': return np.nan
    else: return v

def transform_family(v):
    if v == 'One': return '1'
    elif v == 'Two': return '2'
    elif v == 'Three': return '3'
    elif v == 'Four': return '4'
    elif v == 'Five': return '5'
    elif v == 'Six': return '6'
    elif v == 'Seven': return '7'
    elif v == 'Eight': return '8'
    elif v == 'Nueve': return '9'
    elif v == 'Ten': return '10'
    elif v == 'Once': return '11'
    elif v == 'Twelve or more': return '12'
    elif np.isnan(v): return np.nan

def transform_revenue(v):
    if v == 'less than 1 LMMW': return '1'
    elif v == 'Between 1 and less than 2 LMMW': return '2'
    elif v == 'Between 2 and less than 3 LMMW': return '3'
    elif v == 'Between 3 and less than 5 LMMW': return '4'
    elif v == 'Between 5 and less than 7 LMMW': return '5'
    elif v == 'Between 7 and less than 10 LMMW': return '6'
    elif v == '10 or more LMMW': return '7'
    elif np.isnan(v): return np.nan

def transform_job(v):
    if v == 'No': return '0'
    elif v == 'Yes, less than 20 hours per week': return '1'
    elif v == 'Yes, 20 hours or more per week': return '2'
    elif np.isnan(v): return np.nan

def transform_school(v):
    if v == 'PUBLIC': return 'yes'
    elif v == 'PRIVATE': return 'no'
    elif np.isnan(v): return np.nan

data.JOB = data.JOB.apply(lambda x : transform_job(x))
data.JOB = pd.to_numeric(data.JOB,errors='coerce')

data.SCHOOL_NAT = data.SCHOOL_NAT.apply(lambda x : transform_school(x))
data.STRATUM = data.STRATUM.apply(lambda x : transform_stratum(x))
data.STRATUM = pd.to_numeric(data.STRATUM,errors='coerce')
data.EDU_FATHER = data.EDU_FATHER.apply(lambda x : transform_education(x))
data.EDU_FATHER = pd.to_numeric(data.EDU_FATHER,errors='coerce')
data.EDU_MOTHER = data.EDU_MOTHER.apply(lambda x : transform_education(x))
data.EDU_MOTHER = pd.to_numeric(data.EDU_MOTHER,errors='coerce')
data.SISBEN = data.SISBEN.apply(lambda x : transform_SISBEN(x))
data.PEOPLE_HOUSE = data.PEOPLE_HOUSE.apply(lambda x : transform_family(x))
data.PEOPLE_HOUSE = pd.to_numeric(data.PEOPLE_HOUSE,errors='coerce')
data.REVENUE = data.REVENUE.apply(lambda x : transform_revenue(x))
data.REVENUE = pd.to_numeric(data.REVENUE,errors='coerce')
data = data.rename(columns={"SCHOOL_NAT":"PUBLIC_SCHOOL","STRATUM":"RESIDENCE_STRATUM","SISBEN": "SOCIAL_SUPPORT", "REVENUE": "HOUSEHOLD_INCOME",
                            "FRESH":"REFRIGERATOR", "PEOPLE_HOUSE":"HOUSEHOLD_SIZE"})
data.to_csv('data/application/student_performance_colombia.csv', index = False)
