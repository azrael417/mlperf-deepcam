import subprocess as sp
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import shutil
import sqlite3

def parse_filename_nsight(filename):
    #empty dicts
    result={}
    
    #add network name
    result["Network Name"] = re.match(r'.*\.name_(.*?)\.',filename).groups()[0]
    result["Batch Size"] = int(re.match(r'.*\.batchsize_(.*?)\.',filename).groups()[0])
    result["Input Shape"] = re.match(r'.*\.inputshape_(.*?)\.',filename).groups()[0]
    result["Kernel Shape"] = re.match(r'.*\.kernelshape_(.*?)\.',filename).groups()[0]
    result["Stride Size"] = int(re.match(r'.*\.stride_(.*?)\.',filename).groups()[0])
    result["Data Format"] = re.match(r'.*\.dataformat_(.*?)\.',filename).groups()[0]
    result["Pass"] = re.match(r'.*\.pass_(.*?)\.',filename).groups()[0]
    prec = int(re.match(r'.*\.fp(.*?)\.',filename).groups()[0])
    result["Precision"] = ("FP16" if prec==16 else "FP32")
    
    return result

def parse_filename_nvprof(filename):
    
    #empty dicts
    result={}
    
    #add network name
    result["Network Name"] = re.match(r'.*\.name_(.*?)\.',filename).groups()[0]
    result["Batch Size"] = int(re.match(r'.*\.batchsize_(.*?)\.',filename).groups()[0])
    result["Input Shape"] = re.match(r'.*\.inputshape_(.*?)\.',filename).groups()[0]
    result["Kernel Shape"] = re.match(r'.*\.kernelshape_(.*?)\.',filename).groups()[0]
    result["Stride Size"] = int(re.match(r'.*\.stride_(.*?)\.',filename).groups()[0])
    result["Data Format"] = re.match(r'.*\.dataformat_(.*?)\.',filename).groups()[0]
    result["Pass"] = re.match(r'.*\.pass_(.*?)\.',filename).groups()[0]
    prec = int(re.match(r'.*\.fp(.*?)\.',filename).groups()[0])
    result["Precision"] = "FP16" if prec==16 else "FP32";
    metric = re.match(r'.*\.metric_(.*?)\.',filename).groups()[0]
    
    return result, metric


def import_nvprof_metric(filename, timeline=False, cuda_dir='/usr/local/cuda'):
    #execute nvprof and parse file
    args = [os.path.join(cuda_dir, "bin/nvprof"),"--csv","-i",filename]
    skiprows = 2
    
    #if timeline is enabled, we have to skip less rows also
    if timeline:
        args.append("--print-gpu-trace")
        skiprows = 1
    
    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    #get timeline from csv
    profiledf = pd.read_csv(StringIO(stderr.decode("utf-8")),skiprows=skiprows).dropna(how="all").rename(columns={"Kernel": "Name"})
    profiledf["Collection Type"] = "kernel"
    
    #return result
    return profiledf


def import_nvprof_overview(filename, nvtx=False, cuda_dir='/usr/local/cuda'):
    #execute nvprof and parse file
    args = [os.path.join(cuda_dir, "bin/nvprof"),"--csv","-i",filename]
    
    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()

    #now remove the ranges
    inp = stderr.decode("utf-8")
    
    #get the profiling data
    profile = inp.split("======== NVTX result")[0]
    
    if nvtx:
        marker = inp.split("======== NVTX result")[1]

    #we can readily use the profile info
    profiledf = pd.read_csv(StringIO(profile), skiprows=1, header=[0,1]).dropna(how="all")
    
    #make the time units the same:
    for col in profiledf.columns:
        if col[1] == "ms":
            profiledf[col] *= 10**(-3)
        elif col[1] == "us":
            profiledf[col] *= 10**(-6)
        elif col[1] == "ns":
            profiledf[col] *= 10**(-9)
            
    #now drop that header
    profiledf.columns = profiledf.columns.droplevel(1)
    
    #now sort
    profiledf = profiledf.sort_values(by=["Type", "Name"]).reset_index(drop=True)
    profiledf["Metric Name"] = "time"
    
    #some renamings
    profiledf.loc[ profiledf["Type"] == "GPU activities", "Type" ] = "gpu_activities"
    profiledf.loc[ profiledf["Type"] == "API calls", "Type" ] = "api_calls"
    
    #rename columns
    profiledf.rename(columns={"Type": "Collection Type"}, inplace=True)
    
    if nvtx:
        markerdflist = []
        for it in re.finditer(r"========\s{1,}Range(.*?)(==|$)", marker, flags=re.DOTALL):
            #read into DF
            tmpdf = pd.read_csv(StringIO(it.groups()[0]),skiprows=lambda x: x in [0,2], header=0)
            del tmpdf["Time(%)"]
    
            #drop rows without info
            tmpdf = tmpdf[ ~tmpdf["Type"].str.contains("were profiled in this range") ]
    
            #extract range name:
            rangename = tmpdf.loc[ tmpdf["Type"] == "Range:", "Name" ][0]
        
            #some renamings
            tmpdf.loc[ tmpdf["Type"] == "Range:", "Name" ] = "total"
            tmpdf.loc[ tmpdf["Type"] == "Range:", "Type" ] = "range"
            tmpdf.loc[ tmpdf["Type"] == "GPU activities", "Type" ] = "gpu_activities"
            tmpdf.loc[ tmpdf["Type"] == "API calls", "Type" ] = "api_calls"
    
            #add the rangename to the entries
            tmpdf["Range Name"] = rangename
    
            #renaming
            tmpdf.rename(columns={"Type": "Collection Type"}, inplace=True)
    
            #add to list
            markerdflist.append(tmpdf)
    
        #concat the crap
        markerdf = pd.concat(markerdflist).sort_values(by=["Range Name", "Time"], ascending=[True, False]).reset_index(drop=True)
    else:
        markerdf = pd.DataFrame()
    
    return profiledf, markerdf


def combine_metrics(df, metrics):
    return pd.DataFrame.from_records([{"Metric Count": df[m].values[0], "Metric Name": m.replace("read","").replace("write","").replace("__","_"), \
    "Metric Mode": "read" if "read" in m else "write" if "write" in m else "total"} for m in metrics])


def replace_tc_string(value):
    value = int(re.match(r".*?\((.*?)\)",value).groups()[0])
    return value


def import_nsight_metric(filename, cuda_dir='/usr/local/cuda'):
    #execute nvprof and parse file
    args = [os.path.join(cuda_dir, "bin/nv-nsight-cu-cli"),"--csv","-i",filename]
    #skiprows = 2
        
    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    
    #get timeline from csv
    profiledf = pd.read_csv(StringIO(stdout.decode("utf-8")),skiprows=0) #.dropna(how="all").rename(columns={"Kernel": "Name"})
    
    #clean up
    del profiledf["Process ID"]
    del profiledf["Process Name"]
    del profiledf["Host Name"]
    del profiledf["Kernel Time"]
    del profiledf["Context"]
    del profiledf["Stream"]
    del profiledf["Section Name"]
    
    profiledf = profiledf.groupby(["Kernel Name", "Metric Name"]).apply(lambda x: pd.Series([x["Metric Value"].count(),x["Metric Value"].sum()])).reset_index()
    profiledf.rename(columns={0: "Invocations", 1: "Metric Value", "Kernel Name": "Name"}, inplace=True)
    profiledf['Metric Value'] /=profiledf['Invocations']
    
    #return result
    return profiledf


def import_nsight_overview(filename):
    #execute nvprof and parse file
    conn = sqlite3.connect(filename)
    
    #do the query
    query_cuda = "SELECT end-start,StringIds.value FROM CUPTI_ACTIVITY_KIND_KERNEL INNER JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName = StringIds.id"
    query_cublas = "SELECT end-start,StringIds.value FROM CUBLAS_EVENTS INNER JOIN StringIds ON CUBLAS_EVENTS.nameId = StringIds.id"
    query_cudnn = "SELECT end-start,StringIds.value FROM CUDNN_EVENTS INNER JOIN StringIds ON CUDNN_EVENTS.nameId = StringIds.id"

    #get dataframes
    dflist = []
    #read CUDA stuff
    tmpdf = pd.read_sql_query(query_cuda,conn)
    tmpdf.rename(columns={"value": "Kernel Name", "end-start": "Time"}, inplace=True)
    dflist.append(tmpdf)
    #read cublas stuff
    tmpdf = pd.read_sql_query(query_cublas,conn)
    tmpdf.rename(columns={"value": "Kernel Name", "end-start": "Time"}, inplace=True)
    tmpdf = tmpdf[ ~tmpdf["Kernel Name"].str.contains("cuBLASprofiling started") & ~tmpdf["Kernel Name"].str.contains("cuBLAS finished") ]
    if not tmpdf.empty:
        dflist.append(tmpdf)
    #read cudnn stuff
    tmpdf = pd.read_sql_query(query_cudnn,conn)
    tmpdf.rename(columns={"value": "Kernel Name", "end-start": "Time"}, inplace=True)
    tmpdf = tmpdf[ ~tmpdf["Kernel Name"].str.contains("cuDNNprofiling started") & ~tmpdf["Kernel Name"].str.contains("cuDNN finished") ]
    if not tmpdf.empty:
        dflist.append(tmpdf)
    
    #concat
    timedf = pd.concat(dflist)
    
    #add all the timings together
    tmpdf1 = timedf.groupby("Kernel Name").sum().reset_index()
    tmpdf2 = timedf.groupby("Kernel Name").count().reset_index()
    timedf = tmpdf1.merge(tmpdf2.rename(columns={"Time": "Invocations"}), on="Kernel Name")
    timedf["Time"] *= 1e-6
    timedf["Time Avg"] = timedf["Time"] / timedf["Invocations"]
    timedf.rename(columns={"Kernel Name": "Name"}, inplace=True)
    
    #return result
    return timedf