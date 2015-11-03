#!sqlenv/bin/python

from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import request
import pg
import os

conn_string = "host='192.168.100.200' dbname='hdwu' user='hdwu' password='4321'"
conn = pg.connect(conn_string)

	
app = Flask(__name__)

@app.route('/datasets/trajectory', methods=['GET'])
def list_datasets():
	#conn_string = "host='192.168.100.200' dbname='hdwu' user='hdwu' password='4321'"
	#conn = pg.connect(conn_string)
	query = "select relname as table from pg_stat_user_tables where schemaname = 'trajectory';"
	rows = conn.query(query).getresult()

	datasets = []
	for row in rows:
		datasets.append(row[0])
		
	return jsonify({'datasets': datasets})

@app.route('/datasets/trajectory/test', methods=['GET'])
def retrieve_a_dataset():
	#conn_string = "host='192.168.100.200' dbname='hdwu' user='hdwu' password='4321'"
	#conn = pg.connect(conn_string)
	#if name == "taxi"
	query = "select * from trajectory.test;"
	rows = conn.query(query).getresult()

	trajectory = []
	for row in rows:
		if row[0] == len(trajectory)-1:
			trajectory[row[0]]['points'].append({'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]})

		else: trajectory.append({'tid': row[0], 'points': [{'index': row[1], 'lon': row[2], 'lat': row[3], 'timestamp': row[4]}]})
	if 'tid' in request.args:
		return jsonify({'trajectory': trajectory[int(request.args['tid'])]})
	else: return jsonify({'trajectory': trajectory})
'''
@app.route('dataset/trajectory/test?<tid>')
def retrieve_a_trajectory(tid):
	query = ""

'''
@app.route('/datasets/ROI', methods=['GET'])
def list_ROI():
	query = "select relname as table from pg_stat_user_tables where schemaname = 'roi';"
	rows = conn.query(query).getresult()

	datasets = []
	for row in rows:
		datasets.append(row[0])
		
	return jsonify({'datasets': datasets})

@app.route('/datasets/ROI/<name>', methods=['GET'])
def retrieve_ROI(name):

    parameter = []
    # rid = rid
    if 'rid' in request.args:
        parameter.append('rid = '+request.args['rid'])
   
    # point(lon,lat) @ range
    if 'geo' in request.args:
        parameter.append('point('+request.args['geo']+') @ range')
    
    # range in range
    if 'r' in request.args:
        parameter.append("range @box '"+request.args['r']+"'")
    
    # density
    if 'density' in request.args:
        parameter.append("density >= "+request.args['density'])
    
    if len(parameter) > 0:
        query = "select * from roi."+name+" "+"WHERE "+" AND ".join(parameter)+" ORDER BY rid ASC;"
    else:
        query = "select * from roi."+name+" ORDER BY rid ASC;"
    
    rows = conn.query(query).getresult()
    
    if len(rows) < 1 : abort(400)
    
    dataset = []
    for row in rows:
        rid = row[0]
        density = row[1]
        score = row[2]
        buffer = row[3].split(',')
        range = {'west':buffer[0].strip('(') .strip(')'), 'north': buffer[1].strip('(') .strip(')'), 'east': buffer[2].strip('(') .strip(')'), 'south': buffer[3].strip('(') .strip(')')}
        
        dataset.append({'rid':rid, 'density': density, 'score': score, 'range':range})
    
    return jsonify({'name':name, 'ROI': dataset})
    
'''
add a ROI dataset
{
    name: dataset name
}
'''
@app.route('/datasets/ROI', methods=['POST'])
def creat_roi_dataset():
    if not request.json:
        abort(400)
    name = request.json['name']
    query = "select count(*) as table from pg_stat_user_tables where schemaname = 'roi' and relname = '"+name+"';"
    if conn.query(query).getresult()[0][0] == 1 :
        #return jsonify({'error': 'existed dataset: '+name+''})
        abort(400)
    else:
        query = "CREATE TABLE roi."+name+" (rid bigint, density bigint, score double precision, range box)"
        conn.query(query)
        return jsonify({'OK': 'Create a new ROI dataset: '+ name}),201

@app.route('/datasets/ROI/<string:name>', methods=['POST'])
def insert_roi(name):
    if not request.json or 'range' not in request.json:
        abort(400)
    query = "select count(*) as table from pg_stat_user_tables where schemaname = 'roi' and relname = '"+name+"';"
    if conn.query(query).getresult()[0][0] == 0 :
        return jsonify({'error': 'not exist the dataset: '+name+''}),400
        
    rows = conn.query("select max(rid) from roi."+name+";").getresult()
    
    if rows[0][0] >= 0:
        rid = rows[0][0]+1
    else:
        rid = 0
        
        
    density = request.json.get('density',"NULL")
    score = request.json.get('score',"NULL")
    bound = request.json.get('range',"NULL")
    range = "BOX '"+str(bound['west'])+","+str(bound['north'])+","+str(bound['east'])+","+str(bound['south'])+"'" ##must
    query = "INSERT INTO roi."+name+" VALUES ("+str(rid)+","+str(density)+","+str(score)+","+range+");"
    conn.query(query)
    return jsonify({'OK': 'Add a new ROI to dataset: '+ name}),201

@app.route('/datasets/ROI/<string:name>', methods=['PUT'])
def modify_roi(name):
    if not request.json or 'rid' not in request.json:
        abort(400)
    
    constraint = []
    rid = str(request.json['rid'])
    
    if 'density' in request.json:
        constraint.append("density = "+str(request.json['density']))
    if 'score' in request.json:
        constraint.append("score = "+str(request.json['score']))
    if 'range' in request.json:
        bound = request.json.get('range',"NULL")
        range = "BOX '"+str(bound['west'])+","+str(bound['north'])+","+bound['east']+","+bound['south']+"'"
        constraint.append("range = "+range)
    query = "UPDATE roi."+name+" SET "+",".join(constraint)+" where rid = "+rid+";"
    conn.query(query)

    return jsonify({'OK': 'Update a existing ROI '+rid+' in dataset: '+ name})
        
@app.route('/datasets/ROI/<string:name>', methods=['DELETE'])      
def delete_roi(name):
    if 'rid' in request.args:
    
        query = "select count(*) FROM roi."+name+" WHERE rid = "+request.args['rid']+";"
        if conn.query(query).getresult()[0][0] == 0 : abort(400)
    
        query = "DELETE FROM roi."+name+" WHERE rid = "+request.args['rid']+";"
        conn.query(query)
        return jsonify({'OK': 'Delete a ROI: '+request.args['rid']+' in dataset: '+ name})
    else:
        query = "DROP TABLE roi."+name+";"
        conn.query(query)
        return jsonify({'OK': 'Delete a ROI dataset: '+ name})


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Data not found'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'Query error'}), 500)
    
if __name__ == '__main__':
	app.run(port=5566,debug=True)
