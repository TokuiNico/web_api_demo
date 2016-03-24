# encoding: utf-8

from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import request


import numpy as np

import time, Queue, re, requests, psycopg2cffi, threading
import datetime
from multiprocessing import Process, Pool
# from rtree import index
import logging


# logging.basicConfig(level=logging.DEBUG)

conn_string = "host='127.0.0.1' dbname='XXX' user='XXX' password='XXX'"
conn = psycopg2cffi.connect(conn_string)
	
app = Flask(__name__)


    
'''
@app.route('dataset/trajectory/test?<tid>')
def retrieve_a_trajectory(tid):
	query = ""

'''
@app.route('/datasets/ROI', methods=['GET'])
def list_ROI():
    cur = conn.cursor()
    query = "select relname as table from pg_stat_user_tables where schemaname = 'roi';"
    
    
    cur.execute(query)
    rows = [r for r in cur]
    datasets = []
    for row in rows:
        datasets.append(row[0])

    return jsonify({'datasets': datasets})

@app.route('/datasets/ROI/<name>', methods=['GET'])
def retrieve_ROI(name):
    cur = conn.cursor()
    '''
    r = requests.get("http://127.0.0.1:5566/datasets/ROI")
    
    if name not in r.json()['datasets']:
        return error_request()
    '''
    parameter = []
    # rid = rid
    if 'rid' in request.args:
        if not re.match('[0-9]+',request.args['rid']):
            return error_request()
        
        parameter.append('rid = '+request.args['rid'])

    # point(lon,lat) @ range
    # 測試point是否為有效格式? (須包含兩個值代表經緯度)
    if 'p' in request.args:
        if not re.match('[-+]?[0-9]*\.?[0-9]+\,[-+]?[0-9]*\.?[0-9]+',request.args['p']):
            return error_request()

        parameter.append('point('+request.args['p']+') @ range')
    
    # range in range
    # 測試range是否為有效格式? (須包含四個值代表經緯度範圍
    if 'r' in request.args:
        if not re.match('[-+]?[0-9]*\.?[0-9]+\,[-+]?[0-9]*\.?[0-9]+,[-+]?[0-9]*\.?[0-9]+\,[-+]?[0-9]*\.?[0-9]+',request.args['r']):
            return error_request()
        parameter.append("range @box '"+request.args['r']+"'")

    # density
    if 'd' in request.args:
        if not re.match('[0-9]+',request.args['d']):
            return error_request()
        parameter.append("density >= "+request.args['d'])

    # score
    if 's' in request.args:

        if not re.match('[0-9]*\.?[0-9]+',request.args['s']):
            return error_request()
        parameter.append("score >= "+request.args['s'])
    
    if len(parameter) > 0:
        query = "select * from roi."+name+" "+"WHERE "+" AND ".join(parameter)+" ORDER BY rid ASC;"
    else:
        query = "select * from roi."+name+" ORDER BY rid ASC;"
    
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        conn.rollback()
        cur.close()
        return error_request()

    rows = [r for r in cur]

    dataset = []
    for row in rows:
        rid = row[0]
        density = row[1]
        score = row[2]
        buffer = row[3].split(',')
        range = {'east':buffer[0].strip('(') .strip(')'), 'north': buffer[1].strip('(') .strip(')'), 'west': buffer[2].strip('(') .strip(')'), 'south': buffer[3].strip('(') .strip(')')}
        
        dataset.append({'rid':rid, 'density': density, 'score': score, 'range':range})
    
    return jsonify({'name':name, 'ROI': dataset})

@app.route('/datasets/ROI', methods=['POST'])
def creat_roi_dataset():
    
    if not request.json:
        return ('',400)

    name = request.json['name']
    cur = conn.cursor()
    query = "CREATE TABLE roi."+name+" (rid bigserial , density bigint, score double precision, range box)"

    
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        conn.rollback()
        cur.close()
        return('',400)
    
    conn.commit()
    cur.close()
    return ('',200)

@app.route('/datasets/ROI/<string:name>', methods=['POST'])
def insert_roi(name):
    if not request.json or 'range' not in request.json:
        return('',400)

    density = request.json.get('density',"NULL")
    score = request.json.get('score',"NULL")
    bound = request.json.get('range',"NULL")
    try:
        range = "BOX '"+str(bound['east'])+","+str(bound['north'])+","+str(bound['west'])+","+str(bound['south'])+"'" ##must
    except KeyError: 
        return('',400)
        
    cur = conn.cursor()
    query = "INSERT INTO roi."+name+"(density,score,range) VALUES ("+str(density)+","+str(score)+","+range+");"
    try: 
        cur.execute(query)
    except psycopg2cffi.Error as e:
        conn.rollback()
        cur.close()
        return(e.pgerror,400)
    conn.commit()
    cur.close()
    
    return ('',200)
    
@app.route('/datasets/ROI/<string:name>', methods=['PUT'])
def modify_roi(name):
    if not request.json or 'rid' not in request.json:
        return('',400)
    cur = conn.cursor()
    constraint = []
    rid = str(request.json['rid'])
    
    if 'density' in request.json:
        constraint.append("density = "+str(request.json['density']))
    if 'score' in request.json:
        constraint.append("score = "+str(request.json['score']))
    if 'range' in request.json:
        bound = request.json['range']
        try:
            range = "'"+str(bound['east'])+","+str(bound['north'])+","+str(bound['west'])+","+str(bound['south'])+"'"
        except KeyError: 
            return('',400)
        constraint.append("range = "+range)
    query = "UPDATE roi."+name+" SET "+",".join(constraint)+" where rid = "+rid+";"

    try: 
        cur.execute(query)
        conn.commit()
        cur.close()
    except psycopg2cffi.Error as e:
        conn.rollback()
        cur.close()
        return(e.pgerror,400)

    return('',200)

@app.route('/datasets/ROI/<string:name>', methods=['DELETE'])
def delete_roi(name):

    cur = conn.cursor()
    if 'rid' in request.args:
        
        query = "DELETE FROM roi."+name+" WHERE rid = "+request.args['rid']+";"
        try:
            cur.execute(query)
            conn.commit()
            cur.close()
        except psycopg2cffi.Error as e:
            conn.rollback()
            cur.close()
            return(e.pgerror,400)
        return ("",200)
    else:
        
        query = "DROP TABLE roi."+name+";"
        try:
            cur.execute(query)
            conn.commit()
            cur.close()
        except psycopg2cffi.Error as e:
            conn.rollback()
            cur.close()
            return(e.pgerror,400)
        return ("",200)

#retrieve a dataset/trajectory/~in given range
@app.route('/datasets/trajectory/<name>', methods=['GET'])
def retrieve_a_dataset(name):
    #conn_string = "host='192.168.100.200' dbname='hdwu' user='hdwu' password='4321'"
    #conn = pg.connect(conn_string)
    #if name == "taxi"
    cur = conn.cursor()
    args = request.args

    parameter = []
    trajectory = []
    
    if 'tid' in args:
    
        parameter.append("tid=" + args['tid'])

        
    if 'lon_s'and 'lon_e' and 'lat_s' and 'lat_e' in args:
        
        parameter.append("lon>=" + args['lon_s'] + " and lon<=" + args['lon_e'] + " and lat>=" + args['lat_s'] + " and lat<=" + args['lat_e'])
    
    if 'ts' in args:
        ts = args['ts'].replace('_',' ')
        parameter.append("timestamp >= '" + ts + "'")
        
    if 'te' in args:
        te = args['te'].replace('_',' ')
        parameter.append("timestamp <= '" + te + "'")

    if len(parameter)>0:
        query = "select * from trajectory." + name + " WHERE "+ " AND ".join(parameter) +" ORDER BY tid,index;"
    else: query = "select * from trajectory." + name + " ORDER BY tid,index;"
    
    print query
    
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        conn.rollback()
        cur.close()
        return error_trajectory()

    tid = -1
    rows = [r for r in cur]
    for row in rows:
        # timestamp = row[4].strftime('%Y-%m-%dT%H:%M:%S')
        timestamp = row[4]
        if row[0] != tid:
            trajectory.append({'tid': row[0], 'points': [{'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': timestamp}]})
            tid = row[0]
        else: trajectory[len(trajectory)-1]['points'].append({'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': timestamp})
    return jsonify({'trajectory': trajectory})

@app.route('/algo/density/<name>', methods=['GET'])
def density(name):
    if 'rid' not in request.args:
        jsonify({"density":-1})
    ROI = name
    rid = request.args['rid']
    payload = {'rid':str(rid)}
    r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name+'?rid='+str(rid))
    
    
    if not len(r.json()['ROI']): return jsonify({"density":-1})
    
    if r.json()['name'] == 'error' : return jsonify({"density":-1})
    
    range = r.json()['ROI'][0]['range']
    payload = {'lon_s':str(range['west']), 'lon_e': str(range['east']),
        'lat_s':str(range['south']), 'lat_e': str(range['north'])}
    r = requests.get('http://127.0.0.1:5566/datasets/trajectory/'+name,params=payload)
    
    count = len(r.json()['trajectory'])

    return jsonify({"density":count})
    
@app.route('/algo/tra2seq/<name>', methods=['GET'])
def tra2seq(name):
    #cur = conn.cursor()
    
    if not request.json or 'points' not in request.json:
        return jsonify({'seq':[]})
    
    points = request.json.get("points","NULL");
    
    rid_list = []
    
    range = {'east':-1, 'west':1, 'north':-1, 'south':1}
    
    for point in points:
        lat = point['lat'];
        lon = point['lon'];
        
        # if point in last region, pass
        if float(range['south']) <= float(lat) <= float(range['north']) and float(range['west']) <= float(lon) <= float(range['east']): 
            continue

        
        payload = {'p': str(lon)+","+str(lat)}
        
        r = requests.get("http://127.0.0.1:5566/datasets/ROI/"+name, params=payload)
        
        if not len(r.json()['ROI']): continue
       
        rid = r.json()['ROI'][0]['rid']
        
        density = request.args.get('d',0)
        
        # update range
        range = r.json()['ROI'][0]['range']

        print float(r.json()['ROI'][0]['density'])
        
        #print rid

        if (len(rid_list) == 0 or rid_list[-1] != rid) and float(r.json()['ROI'][0]['density']) >= float(density):
            print "ADD rid " + str(rid)
            rid_list.append(rid)

    return jsonify({'seq':rid_list})
   
@app.route('/algo/deg',methods=['GET'])
def deg():
    if not request.json or 'tr' not in request.json or 'rid' not in request.json:
        return jsonify({'deg':-1})
    tr = request.json['tr']
    rid = request.json['rid']
    r = set(tr[idx+1] for idx,x in enumerate(tr) if x == rid and idx < len(tr)-1)
    return jsonify({'deg':len(r)})
   
'''{}
    input: a list of ROI sequence
    {
        "TRD": [
            [R1,R2,R3...],
            [R10,R11,R12...]
        ]
    }
    
    output: UMG
    {
        "UMG": [
            [0,1,2,3],
            [1,2,3,4],
            [2,3,4,5]
        ]
    }
'''

@app.route('/algo/umg',methods=['GET'])
def umg():
    if not request.json or 'TRD' not in request.json:
        return jsonify({'UMG':[[-1]]})
    
    TRD = request.json['TRD']
    
    #build UMG matrix
    max_rid = 0
    for seq in TRD:
        for roi in seq:
            max_rid = roi if roi > max_rid else max_rid
    UMG = np.zeros((max_rid+1,max_rid+1))
    
    for r1 in range(max_rid+1):
        
        TRsAll = [tr for tr in TRD if (r1 in tr and tr.index(r1)<len(tr)-1)]

        for r2 in range(max_rid+1):
            if r1 == r2: continue
            
            if len(TRsAll)==0: 
                UMG[r1,r2] = 0
                continue

            # get TR that has edge <r1,r2>
            TRs_r1_r2 = [tr for tr in TRsAll if (r1 in tr and r2 in set(tr[idx+1] for idx,x in enumerate(tr) if x == r1 and idx < len(tr)-1))]

            deg_list = []
            
            for tr in TRs_r1_r2:
                payload = {'tr':tr, 'rid': r1}
                r = requests.get('http://127.0.0.1:5566/algo/deg',json = payload)
                deg = r.json()['deg']
                

                deg_list.append(1/float(deg))

            UMG[r1,r2] = float(sum(deg_list))/float(len(TRsAll))
   
    return jsonify({'UMG':UMG.tolist()})
    
@app.route('/algo/score',methods=['GET'])
def ROI_list():
    if not request.json or 'UMG' not in request.json or 'alpha' not in request.json:
        return jsonify({'score':[-1]})
    
    alpha = request.json['alpha']
    if alpha >= 1 or alpha < 0:
        return jsonify({'score':[-1]})
    
    UMG = request.json['UMG']
    size = len(UMG)
    
   
    M = np.asarray(UMG)
    M = np.transpose(M)

    M = M.astype(float)
    
    M = M * alpha
    
    M = np.insert(M,0,1-alpha, axis = 1)
    

    R = np.ones(size+1)
    R_next = np.zeros(size+1)
    
    while not np.array_equal(R_next, R):
        R = R_next
        R_next = np.insert(np.dot(M,R),0,1)
     
    R = np.delete(R_next,0,0)
   
    return jsonify({'score':R.tolist()})
    
@app.route('/algo/AS/<name>', methods=['GET','PUT'])
def AS(name):
    # Prameter:
    #   ts: start time
    #   te: end time
    #   d:  density threshold
    
    # Update region density
    r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name)
    roi_dataset = r.json()['ROI']
    
    for roi in roi_dataset:
        rid = roi['rid']
        
        density = roi['density']
        
        # Only update NULL density
        if not density is None: continue
        payload = {'rid': rid}
        r = requests.get('http://127.0.0.1:5566/algo/density/'+name, params = payload)
        density = r.json()['density']
        payload['density'] = density
        r = requests.put('http://127.0.0.1:5566/datasets/ROI/'+name, json = payload)
    del roi_dataset
    
    
    # get trajectory dataset
    
    parameter = []
    
    
    if 'ts' in request.args:
        parameter.append("timestamp >= '" + request.args['ts'] + "'")
    if 'te' in request.args:
        parameter.append("timestamp <= '" + request.args['te'] + "'")
        
    cur = conn.cursor()
    
    if len(parameter)>0:
        query = "select distinct(tid) from trajectory." + name + " WHERE "+ " AND ".join(parameter) +" ;"
    else: query = "select distinct(tid) from trajectory." + name + " ;"

    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        print query
        conn.rollback()
        cur.close()
        return error_trajectory()
    
    rows = [r for r in cur]
    
    #return jsonify({'tids': rows})
    
    # Convert Tra to TR
    TRD = []
    region_set = set()
    for tids in rows:
        tid = tids[0]
        r = requests.get('http://127.0.0.1:5566/datasets/trajectory/'+name+'?tid='+str(tid))
        

        if len(r.json()['trajectory']) > 0:
            
            trajectory = r.json()['trajectory'][0]
            
            payload = trajectory
            
            payload_params = {}
            if 'd' in request.args:
                payload_params['d'] = request.args['d']
            r = requests.get('http://127.0.0.1:5566/algo/tra2seq/'+name,json = payload, params = payload_params)
            
            try:
                seq = r.json()['seq']
            except ValueError:
                return r.text
                
            if len(seq) > 0:
                for rid in seq:
                    region_set.add(rid)
                TRD.append(seq)
    
    
    r = requests.post('http://127.0.0.1:5566/algo/TRD/'+name)
    
    if len(TRD) == 0:
        return jsonify({'score':-1})
    
    
    # buil rid and serial id map
    region_list = list(region_set)
    del region_set
    
    '''return jsonify({'list':region_list,'TRD': TRD})'''
    
    # replace rid to serial id
    
    for index in range(len(TRD)):
        for i,rid in enumerate(TRD[index]):
            TRD[index][i] = region_list.index(rid)
    
    payload = {"TRD":TRD}
    

    
    r = requests.get('http://127.0.0.1:5566/algo/umg',json = payload)

    
    payload = {"UMG":r.json()["UMG"], "alpha":0.85}
    r = requests.get('http://127.0.0.1:5566/algo/score',json = payload)
    
    # get score of each serial id
    score_list = r.json()['score']
    

    
    # convert serial id to original rid
    score_dict = dict()
    for sid, score in enumerate(score_list):
        try:
            score_dict[region_list[sid]] = score
        except IndexError:
            print len(region_list), len(score_list)
            return jsonify({'score':[-1]})
        
        # Update ROI score
        payload = {'rid': region_list[sid], 'score': score}
        r = requests.put('http://127.0.0.1:5566/datasets/ROI/'+name, json = payload)
        
        
    
    return jsonify({'score': score_dict})
    

@app.route('/algo/DT/<name>', methods=['GET'])
def DT(name):
    if not request.json or 'seq' not in request.json:
        return jsonify({"score":-1})
    
    #Tr.VS
    seq = request.json['seq']
    
    size = len(seq)
    
    score_list = []

    for rid in seq:
        payload = {'rid': rid}
        
        r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name,params = payload)
        
        if len(r.json()['ROI']) > 0:
            score_list.append(r.json()['ROI'][0]['score'])

    score = sum(score_list)

    return jsonify({"score":score/size})

@app.route('/algo/BT/<name>', methods=['GET'])
def BT(name):
    if not request.json or 'seq' not in request.json:
        return jsonify({"score":-1})
    
    #Tr.VS
    seq = request.json['seq']
    
    score_list = []
    
    #計算ROI分數
    for rid in seq:
        payload = {'rid':rid}
        r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name,params = payload)
        if len(r.json()['ROI']) > 0:
            score_list.append(r.json()['ROI'][0]['score'])
    score = sum(score_list)
    
    return jsonify({"score":score})

@app.route('/algo/BTS_old/<name>', methods=['GET'])
def BTS(name):
    # paramater
    #           r=0,0,1,1 MUST
    #           type = "b" or "d" / default 'b'
    #           k = 10
    if 'r' not in request.args:
        jsonify({"trajectory": [-1]})
    type = request.args.get('type','b')
    k = int(request.args.get('k',5))
    r = request.args.get('r')
    density = request.args.get('d',0)
    
    
    tStart = time.time()
    # get roi in range
    payload = {'r': r, 'd':density}
    r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name, params = payload)
    ROIs = r.json()['ROI']
    
    tStop = time.time()
    
    t_roi = tStop - tStart
 
    # build ROI database: ROI_DB
    # build ROI rank list :ROI_rank
    '''
        idx: ROI rtree
        ROI_DB: 紀錄 invert list， rid -> tid1 tid2 tid3... 
        ROI_rank: 紀錄 ROI score，並依分數排列 (rid, score)
    '''
    tStart = time.time()
    
    idx = index.Index()
    ROI_DB = {}
    ROI_rank = []
    for roi in ROIs:
        ROI_DB[roi['rid']]=[]
        ROI_rank.append((roi['rid'],roi['score']))
        
        left, bottom, right, top = (roi['range']['west'],roi['range']['south'],roi['range']['east'],roi['range']['north'])
        idx.insert(int(roi['rid']), (float(left), float(bottom), float(right), float(top)))
        
    
    ROI_rank.sort(key = lambda tup: tup[1],reverse=True) #依照score排序
    
    if len(ROI_rank) ==0: return jsonify({'candidate_tid':[]})
    
    tStop = time.time()
    t_index = tStop - tStart
    
    # get trajectory in range
    cur = conn.cursor()
    
    if not re.match('[-+]?[0-9]*\.?[0-9]+\,[-+]?[0-9]*\.?[0-9]+,[-+]?[0-9]*\.?[0-9]+\,[-+]?[0-9]*\.?[0-9]+',request.args['r']):
        return error_request()
    
    bounds = request.args['r'].split(",")
    east = max(float(bounds[0]), float(bounds[2]))
    west = min(float(bounds[0]), float(bounds[2]))
    north = max(float(bounds[1]), float(bounds[3]))
    south = min(float(bounds[1]), float(bounds[3]))

    tStart = time.time()
    
    query = "select distinct(tid) from trajectory." + name + " where (lon between " + str(west) + " and " + str(east) + ") and (lat between " + str(south) +" and "+ str(north) +");"

    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        print query
        conn.rollback()
        cur.close()
        return error_trajectory()
    
    rows = [r for r in cur]
    
    tStop = time.time()
    t_tid = tStop - tStart
    
    
    # Convert Tra to TR
    TRD = {}

    #return jsonify({'tid': rows})
    tStart = time.time()
    for tids in rows:
        tid = tids[0]
        r = requests.get('http://127.0.0.1:5566/datasets/trajectory/'+name+'?tid='+str(tid))
        

        if len(r.json()['trajectory']) > 0:
            
            trajectory = r.json()['trajectory'][0]
            
            
            
            # fixed density: 5000
            # Todo: 預先算TRD? 刪除ROI?
            '''
            payload_params = {}
            payload_params['d'] = density
            r = requests.get('http://127.0.0.1:5566/algo/tra2seq/'+name,json = trajectory, params = payload_params)
            seq = r.json()['seq']
            '''
            seq = []
            for points in trajectory['points']:
                rid_list = list(idx.intersection((float(points['lon']), float(points['lat']))))
                
                if len(rid_list) == 0:
                    continue
                rid = rid_list[0]
                
                if len(seq) == 0:
                    seq.append(rid)
                elif seq[-1] != rid:
                    seq.append(rid)
            
            print seq
            
            if len(seq) > 0:
                TRD[int(tid)] = seq
                
                temp_DB = []
                
                for rid in seq:
                    temp_DB.append(int(rid))
                for rid in set(temp_DB):
                    ROI_DB[int(rid)].append(int(tid))
    #BTS
    tStop = time.time()
    
    t_TRA = tStop - tStart
    
    
    LBS = float("infinity")
    SBR = ROI_rank[-1]
    
    
    tStart = time.time()
    
    candidate = []
    
    for ROI in ROI_rank:
        if ROI[0] == SBR[0]: break
        
        tid_list = ROI_DB[ROI[0]]
        
        for tid in tid_list:
            '''
            payload = {'seq':TRD[tid]}
            if type == 'd':
                r = requests.get('http://127.0.0.1:5566/algo/DT/'+name,json = payload)
            else: r = requests.get('http://127.0.0.1:5566/algo/BT/'+name,json = payload)

            score = float(r.json()['score'])
            '''
            
            seq = TRD[tid]
            
            size = len(seq)

            score_list = []

            for rid in seq:
            
                ROI_list = [ROI for ROI in ROI_rank if ROI[0]==rid]
                
                if len(ROI_list) > 0:
                    score_list.append(ROI_list[0][1])

            if type == 'd':
                score = sum(score_list)/len(seq)
            else:
                score = sum(score_list)
                
            if (tid, score) not in candidate:
                candidate.append((tid, score))

        candidate.sort(key = lambda tup: tup[1],reverse=True)
        
        if len(candidate) < k:
            LBS = float("infinity")
        else:
            LBS = float(candidate[k-1][1])
        
        
        if type == 'd':
            try:
                SBR = ROI_rank[max( [ i for i in range(len(ROI_rank)) if float(ROI_rank[i][1])>=LBS] )]
            except ValueError:
                SBR = ROI_rank[-1]
        else:
            try:
                SBR = ROI_rank[max([i for i in range(len(ROI_rank)) if sum( [ROI_rank[j][1] for j in range(len(ROI_rank)) if j >= i]) >= LBS])]
            except ValueError:
                SBR = ROI_rank[-1]
    
    tStop = time.time()
    
    t_LBS = tStop - tStart
    
    candidate_set = []
    for candidate_tra in candidate:
        r = requests.get('http://127.0.0.1:5566/datasets/trajectory/'+name+'?tid='+str(candidate_tra[0]))
        if len(r.json()['trajectory']) > 0:
            trajectory = r.json()['trajectory'][0]
            
            trajectory['score'] = candidate_tra[1]
            
            candidate_set.append(trajectory)
            
    return jsonify({"candidate_tid": [ candidate_set[i] for i in range(len(candidate_set)) if i < k], "time":{"ROI":t_roi, "index":t_index, "tid":t_tid, "TRA":t_TRA, "LBS":t_LBS}})
    
def Calculate_TRD_Score(tid_list, ROI_rank, TRD, candidate, type):
    """Calculate score of tid

        1. Get tid in tid_list(in ROI)
        2. Calculate score of tid
        3. Append (tid, score) to candidate
        
    """

    for tid in tid_list:
    # while not tid_que.empty():
        # tid = tid_que.get()
        
        seq = TRD[int(tid)]    # get rid list in TRD
        size = len(seq)
        
        score_list = []
        # lock = threading.Lock()
        
        ROI_list = [ROI for ROI in ROI_rank if ROI[0] in seq]   # list of ROI in seq
        
        for ROI in ROI_list:
            score_list.append(ROI[1])
        
        '''
        for rid in seq:
        
            ROI_list = [ROI for ROI in ROI_rank if ROI[0]==rid]   # search ROI in ROI_rank
            
            if len(ROI_list) > 0:
                score_list.append(ROI_list[0][1])
        '''
        if type == 'd':
            score = sum(score_list)/size
        else:
            score = sum(score_list)
            
        
        #lock.acquire()
        candidate.append((tid, score))
        #lock.release()

def tra2seq2(trajectory_list, idx):
    
    return_list = []
    
    for trajectory in trajectory_list:
        seq = []
        for points in trajectory:
            rid_list = list(idx.intersection((float(points[2]), float(points[3]))))
            if len(rid_list) == 0:continue
            rid = rid_list[0]
            
            if len(seq) == 0:
                seq.append(int(rid))
            elif seq[-1] != rid:
                seq.append(int(rid))
        if len(seq) >0:
            return_list.append(seq)
    return return_list

@app.route('/algo/TRD/<name>', methods = ['POST'])
def update_TRD(name):
    # Mapping trajectory to the sequence of ROIs
    # parameter
    #   d = density
    
    TRD_name = name + "_trd"    
    
    '''
    r = requests.get('http://127.0.0.1:5566/datasets/ROI')
    if TRD_name in r.json()['datasets']:
        return error_request()
    '''
    
    print "get ROI"
    density = request.args.get('d',1)
    payload = {'d':density}
    r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name, params = payload)
    ROIs = r.json()['ROI']
    
    print "build index"
    idx = index.Index()
    for roi in ROIs:
        left, bottom, right, top = (roi['range']['west'],roi['range']['south'],roi['range']['east'],roi['range']['north'])
        idx.insert(int(roi['rid']), (float(left), float(bottom), float(right), float(top)))
    
    cur = conn.cursor()
    
    print "create table: " + TRD_name
    query = "CREATE TABLE roi."+ TRD_name + " (tid bigint, index bigint, rid bigint); CREATE INDEX ON roi."+ TRD_name + " (tid); create index on roi." + TRD_name + " using hash (rid);"
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        print query
        conn.rollback()
        cur.close()
        return jsonify({"ERROR": query})
    
    
    print "get trajectory: " + name
    query = "select tid,index,lon,lat from trajectory." + name + " ORDER BY tid,index;"
    
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        print query
        conn.rollback()
        cur.close()
        return jsonify({"ERROR": query})


    tid = -1
    
    previous_rid = -1
    id = 0
    cur2 = conn.cursor()
    print "insert TRD into: " + TRD_name
    for points in cur:
        #print points[0]
        if tid != points[0]: #complete trajectory
            tid = points[0]
            id = 0

        rid_list = list(idx.intersection((float(points[2]), float(points[3]))))
        if len(rid_list) == 0:continue
        rid = rid_list[0]
        
        if rid != previous_rid:
            query2 = "INSERT INTO roi."+ TRD_name + " VALUES ( " + str(tid) + ", " + str(id) + ", " + str(rid) + " )"

            print query2
            try:
                cur2.execute(query2)
            except psycopg2cffi.Error as e:
                conn.rollback()
                cur2.close()
                return jsonify({"ERROR": query2})
            id += 1
            previous_rid = rid
            
    conn.commit()
    cur2.close()
    cur.close()
    return ("",200)

@app.route('/algo/PATS/<name>', methods=['GET'])
def test3(name):
    # paramater
    #           r=0,0,1,1 MUST
    #           type = "b" or "d" / default 'b'
    #           k = 10
    # if 'r' not in request.args and ('east' not in request.args or 'west' not in request.args or 'north' not in request.args or 'south' not in request.args):
        # jsonify({"trajectory": [-1]})
    
    # if 'r' in request.args:
        # r = request.args.get('r', '-8.701432,41.208771,-8.648355,41.260813')
    # elif 'east' in request.args and 'west' in request.args and 'north' in request.args and 'south' in request.args:
    east = request.args.get('east', '-8.648355')
    west = request.args.get('west', '-8.701432')
    north = request.args.get('north', '41.260813')
    south = request.args.get('south', '41.208771')
    r = str(east)+','+str(north)+','+str(west)+','+str(south)
    
    
    
    type = request.args.get('type','b')
    k = int(request.args.get('k',5))
    
    
    density = request.args.get('d',50000)
    
    
    tStart = time.time()
    # get roi in range
    payload = {'r': r, 'd':density}
    r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name, params = payload)
    ROIs = r.json()['ROI']
    
    tStop = time.time()
    
    t_roi = tStop - tStart
 
    # build ROI database: ROI_DB
    # build ROI rank list :ROI_rank
    '''
        ROI_DB: 紀錄 invert list， rid -> tid1 tid2 tid3... 
        ROI_rank: 紀錄 ROI score，並依分數排列 (rid, score)
    '''
    tStart = time.time()

    ROI_DB = {}
    ROI_rank = []

    for roi in ROIs:
        ROI_DB[roi['rid']]=[]
        ROI_rank.append((roi['rid'],roi['score']))

    ROI_rank.sort(key = lambda tup: tup[1],reverse=True) #依照score排序
    
    if len(ROI_rank) ==0: return jsonify({
        "candidate_tid": [],
        "range":{
            "east": float(east),
            "west": float(west),
            "north": float(north),
            "south": float(south)
        }
        }) 
    
    tStop = time.time()
    t_ROI_DB = tStop - tStart
    
    
######## Convert Tra to TR  ############
    '''
        TRD:
            Save tid: rid1, rid2, rid3...
    '''
    TRD = {}
    
    tStart = time.time()
    
    ROI_list = [str(roi[0]) for roi in ROI_rank]
    
    '''
    
    cur = conn.cursor()
    TRD_name = name + "_trd" 
    query = "SELECT * from roi." + TRD_name + " where rid in (" + ', '.join(ROI_list) + ") ORDER BY tid, index;"
    
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        print query
        conn.rollback()
        cur.close()
        return jsonify({"ERROR": query})
    
    seq = []
    previous_tid = -1
    for points in cur:
        tid, index, rid = points

        if tid != previous_tid: #complete trajectory
            if len(seq)>0:
                TRD[int(tid)] = seq
                for rid in set(seq):
                    ROI_DB[int(rid)].append(int(tid))
            seq = []
            previous_tid = tid

        # if len(seq) == 0:
            # seq.append(int(rid))
        # elif seq[-1] != rid:
            # seq.append(int(rid))
            
        seq.append(int(rid))
    
    
    '''
    import redis

    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    pipeline=r.pipeline()


    for rid in ROI_list:
        # tid_set = r.smembers("rid:"+str(rid))
        # ROI_DB[int(rid)] = list(tid_set)
        tid_set = r.lrange("rid:"+str(rid),0,-1)
        ROI_DB[int(rid)] = tid_set
        
        for tid in tid_set:
            if int(tid) in TRD:
                TRD[int(tid)].append(int(rid))
            else:
                TRD[int(tid)] = [int(rid)]
        

    tid_sets = pipeline.execute()

    
    # '''
    tStop = time.time()
    t_TRA = tStop - tStart
    
    
##############  BTS  ###########################
    print "BTS"

    

    LBS = float("infinity")
    SBR = ROI_rank[-1]

    tStart = time.time()

    candidate = []
    
    complete_tid = set()    # the calculated tid
    
    for ROI in ROI_rank:
        if ROI[0] == SBR[0]:
            logging.info("EARLY STOP")
            break
        
        #tid_list = ROI_DB[ROI[0]]
        logging.info('begin getting tid_list')
        
        candidate_id_set = set(map(lambda can: can[0], candidate))
        
        # tid_list = set(ROI_DB[ROI[0]]) - candidate_id_set
        tid_list = set(ROI_DB[ROI[0]]) - complete_tid #skip calculated tra
        
        complete_tid = complete_tid | set(ROI_DB[ROI[0]]) # add tids in ROI_DB
        
        logging.info('end getting tid_list')
        

        
        logging.info('begin Calculate TRD')
        Calculate_TRD_Score(tid_list, ROI_rank, TRD, candidate,type)
        logging.info('end Calculate TRD')

        
        
        logging.info('begin sort')
        candidate.sort(key = lambda tup: tup[1],reverse=True)
        candidate = candidate[:k]
        
        logging.info('end sort')
        
        
        if len(candidate) < k:
            LBS = float("infinity")
        else:
            LBS = float(candidate[k-1][1])  #LBS = last score of (tid, score) in candidates
        
        logging.info('begin getting SBR')
        if type == 'd':
            try:
                SBR = ROI_rank[max( [ i for i in range(len(ROI_rank)) if float(ROI_rank[i][1])>=LBS] )]
            except ValueError:
                SBR = ROI_rank[-1]
        else:
            try:
                SBR = ROI_rank[max([i for i in range(len(ROI_rank)) if sum( [ROI_rank[j][1] for j in range(len(ROI_rank)) if j >= i]) >= LBS])]
            except ValueError:
                SBR = ROI_rank[-1]

        logging.info('end getting SBR')

    tStop = time.time()
    
    t_LBS = tStop - tStart
    
    candidate_set = []
    
    
    for candidate_tra in candidate:
        r = requests.get('http://127.0.0.1:5566/datasets/trajectory/'+name+'?tid='+str(candidate_tra[0]))
        if len(r.json()['trajectory']) > 0:
            trajectory = r.json()['trajectory'][0]
            
            trajectory['score'] = candidate_tra[1]
            
            candidate_set.append(trajectory)
        
    #return jsonify({"candidate": candidate})
    return jsonify({
            "candidate_tid": [ candidate_set[i] for i in range(len(candidate_set)) if i < k],
            "range":{
                "east": float(east),
                "west": float(west),
                "north": float(north),
                "south": float(south)
            },
            "time":{
                "ROI":t_roi,
                "ROI_DB":t_ROI_DB,
                "TRD": t_TRA,
                "LBS": t_LBS
            }
        }) 
 
@app.route('/algo/show/<name>', methods=['GET'])
def show_algo(name):
    cur = conn.cursor()
    query = "select * from algo_para where algo = '" +  str(name) + "';"
    
    cur.execute(query)
    rows = [r for r in cur]
    para = []
    for row in rows:
        para.append({'name': str(row[1]), 'type': str(row[2]), 'default': float(row[3]), 'min':float(row[4]), 'max': float(row[5])})
    
    
    query = "select * from algo_dataset where algo = '" +  str(name) + "';"
    
    cur.execute(query)
    rows = [r for r in cur]
    dataset = []
    for row in rows:
        dataset.append(str(row[1]))
    
    return jsonify({'ALGO': name, 'Parameter': para, 'Dataset': dataset})

@app.route('/algo/show', methods=['POST'])
def update_algo():
    if not request.json or "ALGO" not in request.json:
        return('',400)
    
    cur = conn.cursor()
    
    name = request.json['ALGO']
    
    
    if 'Parameter' in request.json:
        params = request.json.get('Parameter')
        
        
        for param in params:
            
            defult = param['min']
            
            if 'default' in param:
                defult = param['default']
            
            load = ",".join(["'"+str(name)+"'","'"+str(param['name'])+"'","'"+str(param['type'])+"'",str(defult),str(param['min']),str(param['max'])])
            
            query = "INSERT INTO algo_para VALUES (" + load + ");"
            
            logging.debug(query)
            
            
            try:
                cur.execute(query)
            except psycopg2cffi.Error as e:
                print query
                conn.rollback()
                cur.close()
                return jsonify({"ERROR": query}),400
            
            
            
            
    if 'Dataset' in request.json:
        datasets = request.json.get('Dataset')
        
        
        for dataset in datasets:

            
            query = "INSERT INTO algo_dataset VALUES ('" +name+"','"+ dataset + "');"
            
            logging.debug(query)
            
            try:
                cur.execute(query)
            except psycopg2cffi.Error as e:
                print query
                conn.rollback()
                cur.close()
                return jsonify({"ERROR": query}),400

    conn.commit()
    cur.close()
    return ("",200)
    
def error_request():
    return jsonify({'name':'error','ROI':[{'rid':-1,'density':-1,'score':-1,'range':{'west':'0','east':'0','south':'0','north':'0'}}]})

def error_trajectory():
        return jsonify({'trajectory':[{'points':[{'index':-1,'lat':0,'lon':0}], 'tid':-1}]})
@app.errorhandler(404)
def not_found():
    return make_response(jsonify({'error': 'Not found'}), 404)
    
if __name__ == '__main__':
	app.run(port=5566,threaded=True,debug=True)
