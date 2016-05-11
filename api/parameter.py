from flask import Blueprint, abort, make_response, request, jsonify

import psycopg2cffi
import logging


logging.basicConfig(level=logging.INFO)
conn_string = "host='127.0.0.1' dbname='adsl' user='adsl' password='radsl'"
conn = psycopg2cffi.connect(conn_string)

parameter_page = Blueprint('parameter_page', __name__)

@parameter_page.route('/show/<name>', methods=['GET'])
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

@parameter_page.route('/show', methods=['POST'])
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
 