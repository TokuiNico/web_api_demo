from flask import Blueprint, abort, make_response, request, jsonify

import psycopg2cffi
import logging
import numpy as np
import time, requests

logging.basicConfig(level=logging.INFO)
conn_string = "host='127.0.0.1' dbname='adsl' user='adsl' password='radsl'"
conn = psycopg2cffi.connect(conn_string)

pats_page = Blueprint('pats_page', __name__)

@pats_page.route('/density/<name>', methods=['GET'])
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
    
@pats_page.route('/tra2seq/<name>', methods=['GET'])
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
   
@pats_page.route('/deg',methods=['GET'])
def deg():
    if not request.json or 'tr' not in request.json or 'rid' not in request.json:
        return jsonify({'deg':-1})
    tr = request.json['tr']
    rid = request.json['rid']
    r = set(tr[idx+1] for idx,x in enumerate(tr) if x == rid and idx < len(tr)-1)
    return jsonify({'deg':len(r)})

@pats_page.route('/umg',methods=['GET'])
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
    
@pats_page.route('/score',methods=['GET'])
def Calculate_AS():
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

    
    # M = np.insert(M,0,1-alpha, axis = 1)
    M = np.insert(M,[0],[[1-alpha]] * M.shape[0], axis = 1)
    

    R = np.ones(size+1)
    R_next = np.zeros(size+1)
    
    while not np.array_equal(R_next, R):
        R = R_next
        R_next = np.insert(np.dot(M,R),0,1)
     
    R = np.delete(R_next,0,0)
    
    logging.info(R.tolist())
   
    return jsonify({'score':R.tolist()})

@pats_page.route('/DT/<name>', methods=['GET'])
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

@pats_page.route('/BT/<name>', methods=['GET'])
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
  
def Calculate_TRD_Score(tid_list, ROI_rank, TRD, candidate, type):
    """Calculate score of tid

        1. Get tid in tid_list(in ROI)
        2. Calculate score of tid
        3. Append (tid, score) to candidate
        
    """

    for tid in tid_list:

        seq = TRD[int(tid)]    # get rid list in TRD
        size = len(seq)
        
        score_list = []
        
        ROI_list = [ROI for ROI in ROI_rank if ROI[0] in seq]   # list of ROI in seq

        for ROI in ROI_list:
            score_list.append(ROI[1])

        if type == 'd':
            score = sum(score_list)/size
        else:
            score = sum(score_list)

        candidate.append((tid, score))

@pats_page.route('/TRD/<name>', methods = ['POST'])
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
    density = request.args.get('d',50000)
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

@pats_page.route('/AS/<name>', methods=['GET','PUT'])
def AS(name):
    # Prameter:
    #   ts: start time
    #   te: end time
    #   d:  density threshold
    
    # Update region density
    logging.info("Update region density")
    r = requests.get('http://127.0.0.1:5566/datasets/ROI/'+name)
    roi_dataset = r.json()['ROI']
    
    for roi in roi_dataset: # for all ROIs
        rid = roi['rid']
        density = roi['density']

        if not density is None: continue    #Only update NULL density
        payload = {'rid': rid}
        r = requests.get('http://127.0.0.1:5566/algo/density/'+name, params = payload)  # calculate density of ROI
        density = r.json()['density']
        payload['density'] = density
        r = requests.put('http://127.0.0.1:5566/datasets/ROI/'+name, json = payload)    # update ROI
    del roi_dataset
    
    logging.info("get trajectory tids")
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
    
    logging.info("Convert Tra to TR")
    
    # Convert Tra to TR
    TRD = []
    region_set = set()
    
    logging.info(len(rows))
    
    for tids in rows:
    
        tid = tids[0]
        
        logging.info("tid: "+str(tid))
        
        
        density = request.args.get('d',50000)
        
        query = "select tid, index, rid from trajectory." + name + " as tra left join roi." + name + " as roi on point(tra.lon,tra.lat) @ roi.range where tid=" + str(tid) + " and density >= " + str(density) + " order by index;"
        
        # logging.info(query)
        
        try:
            cur.execute(query)
        except psycopg2cffi.Error as e:
            
            logging.error("error sql: "+query)
        
            conn.rollback()
            cur.close()
            return error_request()
            
        seq = []
        
        # [r for r in cur]
        
        
        for point in cur:
            # logging.info(point)
            
            tid, index, rid = point
            
            region_set.add(rid)
            
            if len(seq) == 0:
                seq.append(rid)
            elif seq[-1] != rid:
                seq.append(rid)
        
        logging.info(seq)

        
        if len(seq) > 0:
            TRD.append(seq)
        
        '''
        r = requests.get('http://127.0.0.1:5566/datasets/trajectory/'+name+'?tid='+str(tid))
        

        if len(r.json()['trajectory']) > 0:
            
            trajectory = r.json()['trajectory'][0]
            
            payload = trajectory
            
            payload_params = {}
            if 'd' in request.args:
                # density = request.args.get('d',50000)
                payload_params['d'] = request.args.get('d',50000)
            r = requests.get('http://127.0.0.1:5566/algo/tra2seq/'+name,json = payload, params = payload_params)
            
            try:
                seq = r.json()['seq']
            except ValueError:
                return r.text
                
            if len(seq) > 0:
                for rid in seq:
                    region_set.add(rid)
                TRD.append(seq)
    
        '''
    
    # r = requests.post('http://127.0.0.1:5566/algo/TRD/'+name)
    
    if len(TRD) == 0:
        return jsonify({'score':-1})
    
    # buil rid and serial id map
    region_list = list(region_set)
    del region_set
    
    '''return jsonify({'list':region_list,'TRD': TRD})'''
    
    # replace rid to serial id
    
    for index in range(len(TRD)):
        for i,rid in enumerate(TRD[index]):
            TRD[index][i] = region_list.index(rid)  #Trasfer RID to RID index(to decrease the size of umg)
    
    payload = {"TRD":TRD}
    
    
    logging.info("UMG")

    
    r = requests.get('http://127.0.0.1:5566/algo/umg',json = payload)

    
    logging.info("Calculate Score")
    payload = {"UMG":r.json()["UMG"], "alpha":0.85}
    

    r = requests.get('http://127.0.0.1:5566/algo/score',json = payload)
    
    # get score of each serial id
    
    # return jsonify({'test':0})
    
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
    
@pats_page.route('/PATS/<name>', methods=['GET'])
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

        
def error_request():
    return jsonify({'name':'error','ROI':[{'rid':-1,'density':-1,'score':-1,'range':{'west':'0','east':'0','south':'0','north':'0'}}]})

def error_trajectory():
        return jsonify({'trajectory':[{'points':[{'index':-1,'lat':0,'lon':0}], 'tid':-1}]})