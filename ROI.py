# encoding: utf-8

from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import request
import psycopg2
import requests
import re
import numpy as np

conn_string = "host='127.0.0.1' dbname='adsl' user='adsl' password='radsl'"
conn = psycopg2.connect(conn_string)
	
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
    except psycopg2.Error as e:
        print "error:"+query
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
    except psycopg2.Error as e:
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
    except psycopg2.Error as e:
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
    except psycopg2.Error as e:
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
        except psycopg2.Error as e:
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
        except psycopg2.Error as e:
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
    trajectory = []
    if 'tid' in args:
        query = "select * from trajectory." + name + " where tid=" + args['tid']+ ";"
        try:
            cur.execute(query)
        except psycopg2.Error as e:
            conn.rollback()
            cur.close()
            return error_request()

        rows = [r for r in cur]
        for row in rows:
            if len(trajectory) == 0:
                trajectory.append({'tid': row[0], 'points': [{'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]}]})
            else: trajectory[0]['points'].append({'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]})

    elif 'lon_s'and 'lon_e' and 'lat_s' and 'lat_e' in args:
        query = "select * from trajectory." + name + " where lon>=" + args['lon_s'] + " and lon<=" + args['lon_e'] + " and lat>=" + args['lat_s'] + " and lat<=" + args['lat_e'] + ";"
        try:
            cur.execute(query)
        except psycopg2.Error as e:
            conn.rollback()
            cur.close()
            return error_request()

        rows = [r for r in cur]
        for row in rows:
            print row
            if row[0] == len(trajectory)-1:                
                trajectory[row[0]]['points'].append({'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]})
            else: trajectory.append({'tid': row[0], 'points': [{'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]}]})

    else:
        query = "select * from trajectory." + name + ";"
        try:
            cur.execute(query)
        except psycopg2.Error as e:
            conn.rollback()
            cur.close()
            return error_request()

        rows = [r for r in cur]
        for row in rows:
            if row[0] == len(trajectory)-1:
                trajectory[row[0]]['points'].append({'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]})
            else: trajectory.append({'tid': row[0], 'points': [{'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]}]})
    return jsonify({'trajectory': trajectory})

@app.route('/algo/tra2seq/<name>', methods=['GET'])
def tra2seq(name):
    #cur = conn.cursor()
    
    if not request.json or 'points' not in request.json:
        return jsonify({'rid':[-1]})
    
    points = request.json.get("points","NULL");
    
    rid_list = []
    
    for point in points:
        lat = point['lat'];
        lon = point['lon'];
        
        if 'd' in request.args:
            url = "http://127.0.0.1:5566/datasets/ROI/"+name+"?p="+str(lon)+","+str(lat)+"&d="+str(request.args['d'])
        else:
            url = "http://127.0.0.1:5566/datasets/ROI/"+name+"?p="+str(lon)+","+str(lat)
        
        r = requests.get(url)
        
        if not len(r.json()['ROI']): continue
        
        rid = r.json()['ROI'][0]['rid']
        
        print rid
        
        if(len(rid_list) == 0):
            rid_list.append(rid)
        elif(rid_list[-1] != rid):
            rid_list.append(rid)

    return jsonify({'seq':rid_list})
   
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
'''
@app.route('/algo/umg',methods=['GET'])
def umg():
    if not request.json or 'TRD' not in request.json:
        return jsonify({'UMG':[]})
    datasets = request.json['TRD']
    
    
    max_rid = 0
    for seq in datasets:
        for roi in seq:
            max_rid = roi if roi > max_rid else max_rid
    umg = np.zeros(max_rid,max_rid)
'''    
        
def error_request():
    return jsonify({'name':'error','ROI':[{'rid':-1,'density':-1,'score':-1,'range':{'west':'0','east':'0','south':'0','north':'0'}}]})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
    
if __name__ == '__main__':
	app.run(port=5566,threaded=True,debug=True)
