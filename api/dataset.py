from flask import Blueprint, abort, make_response, request, jsonify

import psycopg2cffi
import logging
import re
import requests

from splitRects import splitRects

logging.basicConfig(level=logging.INFO)
conn_string = "host='127.0.0.1' dbname='adsl' user='adsl' password='radsl'"
conn = psycopg2cffi.connect(conn_string)

dataset_page = Blueprint('dataset_page', __name__)

@dataset_page.route('/ROI')
def list_ROI():
    cur = conn.cursor()
    query = "select relname as table from pg_stat_user_tables where schemaname = 'roi';"
    
    
    cur.execute(query)
    rows = [r for r in cur]
    datasets = []
    for row in rows:
        datasets.append(row[0])

    return jsonify({'datasets': datasets})
    
@dataset_page.route('/ROI/<name>', methods=['GET'])
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

@dataset_page.route('/ROI', methods=['POST'])
def creat_roi_dataset():
    
    if not request.json:
        return ('no json object',400)

    name = request.json['name']
    cur = conn.cursor()
    query = "CREATE TABLE roi."+name+" (rid bigserial , density bigint, score double precision, range box);"

    
    try:
        cur.execute(query)
    except psycopg2cffi.Error as e:
        logging.error(query)
        conn.rollback()
        cur.close()
        return('Unable to create roi dataset\n',400)
    
    conn.commit()
    
    if 'range' in request.json and 'size' in request.json:
        if 'south' in request.json['range'] and 'north' in request.json['range'] and 'west' in request.json['range'] and 'east' in request.json['range']:
            Sbound = request.json['range']['south']
            Nbound = request.json['range']['north']
            Wbound = request.json['range']['west']
            Ebound = request.json['range']['east']
            size = request.json['size']
            ROIs = splitRects(Sbound, Wbound, Nbound, Ebound, size)
            
            range_list = []
            for ROI in ROIs:
                payload = {'range': ROI}
                r = requests.post('http://127.0.0.1:5566/datasets/ROI/'+name,json = payload)
        
    cur.close()
    return ('',200)

@dataset_page.route('/ROI/<string:name>', methods=['POST'])
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
    
@dataset_page.route('/ROI/<string:name>', methods=['PUT'])
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

@dataset_page.route('/ROI/<string:name>', methods=['DELETE'])
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

@dataset_page.route('/trajectory/<name>', methods=['GET'])
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
    
def error_request():
    return jsonify({'name':'error','ROI':[{'rid':-1,'density':-1,'score':-1,'range':{'west':'0','east':'0','south':'0','north':'0'}}]})

def error_trajectory():
        return jsonify({'trajectory':[{'points':[{'index':-1,'lat':0,'lon':0}], 'tid':-1}]})