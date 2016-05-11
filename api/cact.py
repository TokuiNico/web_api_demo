#!flask/bin/python
import psycopg2
import math
import timeit
from math import sqrt, pow
from operator import itemgetter, attrgetter  
import networkx as nx
from flask import Flask,jsonify,request, Blueprint
import urllib2, json,copy

cact_page = Blueprint('cact_page', __name__)

deta = 0.5

conn_string = "host='localhost' dbname='adsl' user='adsl' port='5432' password='radsl'"
conn = psycopg2.connect(conn_string)
# print "Opened pg database successfully"

@cact_page.route('/api/cact/<string:name>', methods=['GET'])
def getFileSimilarity(name): # A file similarity (may be trajectories in range?)
    start = timeit.default_timer()
    args = request.args
    query=""
    if 'tid1' and 'tid2' in args:
        tid1 = args['tid1']
        tid2 = args['tid2']
        query = "SELECT * From cact.trajectory_artifical where tid>=" + tid1 + " and tid<=" + tid2 + ";"
    elif 'lon1' and 'lon2' and 'lat1' and 'lat2' in args:
        lon1 = args['lon1']
        lon2 = args['lon2']
        lat1 = args['lat1']
        lat2 = args['lat2']
        query = "SELECT * From cact.trajectory_artifical where lat>=" + lat1 + " and lat<=" + lat2 + " and lon>=" + lon1 + " and lon<=" + lon2 + ";"
        
    G = nx.Graph()
    cur = conn.cursor()

    
    min_sup = 4
    spatialThres = 4
    temporalThres = 4
    AllTrajectoriesList = []
    coresetIndex=[]
    AllTrajSim=[]
    Clist =[]
    label = []
    CSEP=[]
    valuei=[]

    #get trajectory from db
    
 
 
    cur.execute(query)
    rows = cur.fetchall()
    
    last_id = rows[0][0] ;
    trajectoryPointList=[];
    newtid = 0;

    for row in rows:
    	
      	tid = int(row[0])
      	lon = row[1]
      	lat = row[2]
      	time = row[3]

      	if(tid==last_id):
      	  p = Point(float(lon),float(lat),int(time))
      	  trajectoryPointList.append(p)
	else:
	    #print "new traj"
	    
	    AllTrajectoriesList.append(trajectoryPointList)
	    
	 #   G.add_node(last_id)
	    #coresetIndex.append(int(last_id))
            G.add_node(newtid)
	    coresetIndex.append(int(newtid))

	    newtid+=1
	    last_id = tid
	    trajectoryPointList = []

	    p = Point(float(lon),float(lat),int(time))
	    trajectoryPointList.append(p)

    
    AllTrajectoriesList.append(trajectoryPointList)
#    G.add_node(last_id)
 #   coresetIndex.append(last_id)
    G.add_node(newtid)
    coresetIndex.append(newtid)
    conn.commit
   

    #calculate trajectories simlarity

    start_sim = timeit.default_timer()              
    for index1 in range(0,len(AllTrajectoriesList),+1):
        traj1 = AllTrajectoriesList[index1]
        similarityList = []
        for index2 in range(0,len(AllTrajectoriesList),+1):
            #print str(index1) + "," + str(index2)
            traj2 = AllTrajectoriesList[index2]
            sim  = 0.0
            sim = getSimilarityValue(traj1,traj2)
            similarityList.append(sim)

        AllTrajSim.append(similarityList)

    stop = timeit.default_timer()
    print "simlarity run time(s) : " + str(stop - start_sim)

    #return str(AllTrajSim)
    
    #print "Similarity : " + str(AllTrajSim)


    #add edge which simalarity>deta
    for i in range (0, len(coresetIndex),+1):
        for j in range(0, len(coresetIndex),+1):
            if(i!=j and float(AllTrajSim[i][j])>=deta and float(AllTrajSim[j][i])>=deta):
                G.add_edge(i,j)

    #print "i:" + str(i) + ", j:" + str(j)



    #core set identify  , change (using graph find clique)
                
#    CoresetList=[]

#    for index1 in range(0,len(coresetIndex),+1):
#        coreset = []
#        coreset.append(index1)
#        for index2 in range(0,len(coresetIndex),+1):
#            if(index1!=index2):
#                for core in coreset: # check with coreset 2 direction

#                    if(float(AllTrajSim[core][index2])>=deta and float(AllTrajSim[index2][core])>=deta):
#                        G.add_edge(core,index2)
#                        isCore = True
#                    else:
#                        isCore = False
#                        break
#                if isCore==True:# all pass
#                    coreset.append(index2)
#        CoresetList.append(coreset)


    #print "num of nodes :" + str(G.number_of_nodes())
    #print "edges:" + str(G.edges())




    #sort all cliques with length , remove the largest one  and so on
    
    G_copy = G
    final_coreset = []
    cliquelist = list(nx.find_cliques(G))
    cliquelist.sort(key = len,  reverse=True)

    while(len(cliquelist)>0):
        final_coreset.append(cliquelist[0])
        #print "remove : " + str(cliquelist[0])
        for each in cliquelist[0]:
            G.remove_node(each)


        cliquelist = list(nx.find_cliques(G))
        cliquelist.sort(key = len,  reverse=True)#sort all cliques


    stop = timeit.default_timer()
    print "find all cliques run time(s) : " + str(stop - start)
    
    
    #print "coreset : " + str(final_coreset)


    
    CoresetList3 = final_coreset

    

#algo line 7 to 9


    Klist=[]
    COOH=[]
    

#init

    index=0
    for coreset in final_coreset:

        Clist.append(coreset)#clusters after find clique
        k=[]
        k.append(index)
        Klist.append(k) # which C in k
        COOH.append(0)
        CSEPlist = []
        ilist=[]
        
        for i in range(0,len(CoresetList3),+1):
            CSEPlist.append(0.00)
            ilist.append(0.00)
        CSEP.append(CSEPlist)
        valuei.append(ilist)

        index+=1

    
    print "clist:" + str(Clist)                     
#count km to kn total weights
    for i in range(0,len(Klist),+1):
        for j in range(0,len(Klist),+1):
            if(i!=j):
                total = 0.0
                coreset1 = Klist[i][0]
                coreset2 = Klist[j][0]
                noedge= 0
                
                for index1 in Clist[int(coreset1)]: # index is Traj no.
                    hasedge = False
                    for index2 in Clist[int(coreset2)]:
                        #print str(index1) + "," + str(index2)
                        if(int(index1)!=int(index2)):
                            print str(index1) +"," + str(index2)
                            if (float(AllTrajSim[int(index1)][int(index2)])>=deta):
                               total += float(AllTrajSim[int(index1)][int(index2)])
                               hasedge = True
                    if(hasedge==False):
                        noedge+=1
                    
                            
                CSEP[i][j] = total # i,j is C no.
                            
                
            else:
                CSEP[i][j] = 0
            


    done=False
    while(done==False):
        benefit = 0.0
        r = -1.0
        merge = []
        for i in range(0,len(Klist),+1):
            
            
            for j in range(0,len(Klist),+1):
                if(i!=j):
                    des = (CSEP[i][j] + CSEP[j][i])/2                   
                    inc = 0.0
                    ilist= []
                    ilist.append(Klist[i])
                    ilist.append(Klist[j])

                    q = getI(ilist,Clist,AllTrajSim)*deta
                    inc = getCOOH(Klist[i],Clist,AllTrajSim) + getCOOH(Klist[j],Clist,AllTrajSim) + q
                    benefit = des - inc

                    if(benefit>r):
                        merge = []
                        merge.append(int(i))
                        merge.append(int(j))
                        r = benefit

        
        if(r>0 and len(merge)>1):
            newlist = []
            for ci in Klist[int(merge[0])]:#km
                newlist.append(ci)
            for cj in Klist[int(merge[1])]:#kn
                newlist.append(cj)

            if(int(merge[0])>int(merge[1])):
                Klist.pop(int(merge[0]))
                Klist.pop(int(merge[1]))
            else:
                Klist.pop(int(merge[1]))
                Klist.pop(int(merge[0]))

            Klist.append(newlist)
            
        else:
            
            done = True
                    
        
        CSEP = updateCSEP(Klist,CSEP,Clist,AllTrajSim)
        
        result = []
        
        for k in Klist:
            newlist = []
            for c in k:
                for v in Clist[int(c)]:
                    if(v not in newlist):                       
                        newlist.append(v)
            result.append(newlist)
        
        

    result = []
           

    stop = timeit.default_timer()
    print "Algo run time(s) : " + str(stop - start)
    # Algo 3

    finalTrajList=[]
    for k in range(0, len(Klist) , +1):

        for c in range(0, len(Klist[k]), +1):
            RP = []
            PSi = []
            
            cindex = Klist[k][c]# core set index
            ci = Clist[cindex]

            #print str(ci)
            highestsum = 0.0


            if(len(ci)!=1): # not only 1 traj in coreset
                for v in ci:
                    temptotal = 0.0

                    for i in range (0, len(coresetIndex),+1):
                        if(i!=v and float(AllTrajSim[i][v])>=deta):
                            temptotal+=float(AllTrajSim[i][v])
                            

                    if temptotal>highestsum:
                        highestsum = temptotal
                        BS = v

                #base traj in this coreset
                
                PSi = AllTrajectoriesList[BS]

                if(True):
                    for each in range(0, len(ci),+1): # each traj in coreset
        
                        t = ci[each]
                        if(t!=BS):
                            pointList = AllTrajectoriesList[t]

                            if(True):
                                  for p in pointList:
                                      
                                      ps =0
                                      pe =0

                                      mindistance = 1000
                                      # this point is close to which line
                                      for index in range(0, len(PSi)-1, +1):
                                          if(p.time>(PSi[index+1].time + temporalThres)):# this point over constraint
                                              break
                                          
                                          if(p.time>=(PSi[index].time - temporalThres) and p.time<=(PSi[index+1].time + temporalThres)):
                                              distance = DistancePointLine(p.lon,p.lat,PSi[index].lon,PSi[index].lat,PSi[index+1].lon,PSi[index+1].lat)

                                              distance = distance*100000
                                              #print distance
                                              if(distance<mindistance):
                                                  mindistance = distance
                                                  ps = index
                                                  pe = index+1

                                            #30 m 
                                              if(mindistance<200 and p.time>=(PSi[ps].time - temporalThres) and p.time<=(PSi[pe].time + temporalThres)):
                                                  if(EuclideanDistance(PSi[ps],p)!=0 and EuclideanDistance(p,PSi[pe])!=0):
                                                      pt = PSi[ps].time + (PSi[pe].time - PSi[ps].time)* (EuclideanDistance(PSi[ps],p)/(EuclideanDistance(PSi[ps],p)+EuclideanDistance(p,PSi[pe])))
                                                      new_p = Point(float(p.lon),float(p.lat),float(pt))
                                                      PSi.insert(ps+1,new_p)
                                                      #print "insert"

                vecPoint = PSi
                #Create object  
                dbScan = DBSCAN()  
                #Load data into object  
                dbScan.DB = vecPoint;  
                #Do clustering  
                dbScan.DBSCAN()  
                #Show result cluster

                RPi = []
                
                for i in range(len(dbScan.cluster)):  
 #                 print 'Cluster: ', i
                  lon=0.0
                  lat=0.0
                  for j in range(len(dbScan.cluster[i])):
                      
                      lon += dbScan.cluster[i][j].lon
                      lat += dbScan.cluster[i][j].lat

                  lon = lon/len(dbScan.cluster[i])
                  lat = lat/len(dbScan.cluster[i])

                  p = Point(lon,lat,0.0)
                  RPi.append(p)


                for i in range(len(dbScan.noise)):  

                  RPi.append(dbScan.noise[i])

                
            else:# if core set only 1 element
                PSi = AllTrajectoriesList[ci[0]]
                PC = PSi # only one traj in coreset = PC (dbscan)

                PRi = PC


            RP.append(RPi)
                
            traj = []
            
            for i in RP:
                for each in i:
                    #print each.show()
                    traj.append([each.lon,each.lat])

            finalTrajList.append(traj)
                               

    print "total run time(s) : " + str(stop - start)
    tid = 0
    jsonresult = []
    for traj in finalTrajList:
        t=[]
        for point in traj:
            p = {"lon":point[0],"lat":point[1]}
            t.append(p)
        tj = { "tid":tid,"points": t}
        tid+=1
        jsonresult.append(tj)
        
    
    return jsonify({"Trajectory":jsonresult})


@cact_page.route('/todo/api/v1.0/tasks/CATS/getSimilarity/<int:trajectory_id1>&<int:trajectory_id2>', methods=['GET'])
def getSimilarity(trajectory_id1,trajectory_id2): #two trajectories T1 to T2 similarity

        
    file1 = open('/Users/annaiam/app/'+str(trajectory_id1) + '.txt') #select from DB trajectoryID
    file2 = open('/Users/annaiam/app/'+str(trajectory_id2) + '.txt') #select from DB trajectoryID

    spatialThres = 0.5414
 #   temporalThres = 4
    temporalThres = 4
    trajectoryList1 = []
    trajectoryList2 = []
    TotalScore = 0.000
    
    for line in file1:
        string = str(line.strip())
        trajectory = string.split(',')
        p = PointGPS(float(trajectory[0]),float(trajectory[1]),int(trajectory[2]))
        trajectoryList1.append(p)

    #print str(p.lon)


    for line in file2:
        string = str(line.strip())
        trajectory = string.split(',')
        p = Point(float(trajectory[0]),float(trajectory[1]),int(trajectory[2]))
        trajectoryList2.append(p)
       

    for point1 in trajectoryList1:
        
        clueScore = 0
        for point2 in trajectoryList2:
            #print "test"
            if(abs(int(point1.time)-int(point2.time)) <= temporalThres):
                clueScore = getMaxScore(clueScore, SpatialDecaying(spatialThres,point1,point2))
                #print str(clueScore)
        TotalScore += clueScore;

    #print(TotalScore/len(trajectoryList1))
    
    #return str(TotalScore/len(trajectoryList1))
    return jsonify({'similarity': TotalScore/len(trajectoryList1)})

def getMaxScore(score1,score2):
    if(score1>=score2):
        return score1
    else:
        return score2


def SpatialDecaying(spatialThres,point1,point2):
    #consider spatialThres as 4
    #point1 = Point(7,4,9)
    #point2 = Point(3,3,3)
    dist = EuclideanDistance(point1,point2)
    if(dist>spatialThres):
        return 0
    else:
        return (1-dist/spatialThres)
    

def EuclideanDistance(point1,point2):
 #   dist = ((point1.lon-point2.lon)**2 + (point1.lat-point2.lat)**2)**0.5
    radius = 6371 # km

    dlat = math.radians(point2.lat-point1.lat)
    dlon = math.radians(point2.lon-point1.lon)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(point1.lat)) \
         * math.cos(math.radians(point2.lat)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

     #print d
    return d*100
#    return dist

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km




def getSimilarityValue(trajectoryList1,trajectoryList2): #two trajectories T1 to T2 similarity

    spatialThres = 4
    temporalThres = 4
    TotalScore = 0.000

    for point1 in trajectoryList1:
        
        clueScore = 0
        for point2 in trajectoryList2:
            
            if(abs(point1.time-point2.time) <= temporalThres):
                clueScore = getMaxScore(clueScore, SpatialDecaying(spatialThres,point1,point2))
        TotalScore += clueScore;

    #print(TotalScore/len(trajectoryList1))
    
    return str(TotalScore/len(trajectoryList1))

class Point:
    lon =0
    lat =0
    time=0
    visited = False  
    isnoise = False 
                  
    def __init__(self, lon, lat, time , visited = False, isnoise = False):
        self.lon = lon
        self.lat = lat
        self.time = time
        self.visited = False  
        self.isnoise = False 
 
   
    def show(self):
        return self.lat, self.lon,self.time 


def updateCSEP(list1,CSEP,Clist,AllTrajSim):

    
    for i in range(0,len(list1),+1):
        for j in range(0,len(list1),+1):
            if(i!=j):
                total = 0.0
                coreset1 = list1[i]
                coreset2 = list1[j]
                for ci in coreset1:
                    for cj in coreset2:
                        for index1 in Clist[int(ci)]:
                            for index2 in Clist[int(cj)]:
                                if(int(index1)!=int(index2) and float(AllTrajSim[int(index1)][int(index2)])>=deta):
                                    total += float(AllTrajSim[int(index1)][int(index2)])
                                               
                CSEP[i][j] = total
                            
                
            else:
                CSEP[i][j] = 0
    return CSEP    

def getCOOH(list1,Clist,AllTrajSim):#Klist contain clusters no.
    minedgeCount = 100000.0
    if(len(list1)==1): # only 1 cluster
        return 0
    else:
        for i in range(0,len(list1),+1): # assum i as core set
            score = 0.0
            ci = list1[i]
            for j in range(0, len(list1),+1):#count all Cj to Ci need to add how much
                if(i!=j):
                    cj = list1[j]
#                    print str(ci) + "to" + str(cj)
                    for eachj in Clist[cj]: # each = Traj
                        ejToei = False
                        for eachi in Clist[ci]:
#                            print str(eachi) + "," +str(eachj)
                            if(float(AllTrajSim[int(eachj)][int(eachi)])>=deta):
                                ejToei = True# if this Traj in j to Ci
#                                print "edge" + str(eachj) + " " + str(eachi)
                                break
                        if(ejToei==False):
                            score +=1.0
                            #print "add"
#            print "assum core = " +  str(ci) + "," + "score" + str(score)
            minedgeCount = min( minedgeCount, score)
#        print str(list1) + "COOH" + str(minedgeCount*deta) 
                            
                        
        
#        for i in range(0,len(list1),+1):
#            for j in range(i, len(list1),+1):
#                if(i!=j):
#                    ci = list1[i]
#                    cj = list1[j]
#                    ciTocj = valuei[int(ci)][int(cj)]
#                    cjToci = valuei[int(cj)][int(ci)]
                    
#                    score+= min( ciTocj, cjToci)                 
            
        return minedgeCount*deta

def getI(list1,Clist,AllTrajSim):#Klist contain 2 k
    k1 = list1[0]
    k2 = list1[1]
    total = 0.0

    for i in range(0,len(k1),+1): # assum i as Ci
            minscore = 100000.0
            ci = k1[i] # core set ci in km
            for j in range(0, len(k2),+1):#count all Ci to Cj needs
                    valueI = 0.0
                    cj = k2[j]
                    #print "Clist" + str(Clist)
                    #print str(ci)
                    #print str(len(Clist))
                    for eachi in Clist[ci]: # each = Traj i is connect to j?
                        eiToej = False
                        for eachj in Clist[cj]:
                            #print str(eachi) + "," +str(eachj)
                            if(float(AllTrajSim[int(eachi)][int(eachj)])>=deta):
                                eiToej = True# if this Traj in j to Ci
                                #print "edge" + str(eachj) + " " + str(eachi)
                                break
                        if(eiToej==False):
                            valueI +=1.0
                            #print "add"
                    #print "i:" + str(valueI) +", min compare with" + str(minscore) 
                    minscore = min(minscore,valueI)
            total+=minscore
    return total


def equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for each in list2:
        if each in list1:
            continue
        else:
            return False
    return True

def include(list1,list2):
    if len(list1) > len(list2):
        for each in list2:
            if each in list1:
                continue
            else:
                return False
        return True
    else:
        for each in list1:
            if each in list2:
                continue
            else:
                return False
        return True
    
def overlap(list1,list2):
    overlaplist=[]
    if len(list1) > len(list2):
        for each in list2:
            if each in list1:
                overlaplist.append(each)
                continue
        return overlaplist
    else:
        for each in list1:
            if each in list2:
                overlaplist.append(each)
                continue
        return overlaplist

    return True


def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude
 
#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine (px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = math.hypot(x2-x1, y2-y1)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine


class DBSCAN:  
 #Density-Based Spatial Clustering of Application with Noise -> http://en.wikipedia.org/wiki/DBSCAN  
   def __init__(self):  
     self.name = 'DBSCAN'  
     self.DB = [] #Database  
     self.esp = 4 #neighborhood distance for search  
     self.MinPts = 2 #minimum number of points required to form a cluster  
     self.cluster_inx = -1  
     self.cluster = []
     self.noise = []
       
   def DBSCAN(self):  
     for i in range(len(self.DB)):  
       p_tmp = self.DB[i]  
       if (not p_tmp.visited):  
         #for each unvisited point P in dataset  
         p_tmp.visited = True  
         NeighborPts = self.regionQuery(p_tmp)  
         if(len(NeighborPts) < self.MinPts):  
           #that point is a noise  
           p_tmp.isnoise = True  
 #          print p_tmp.show(), 'is a noise'
           self.noise.append(p_tmp)
         else:  
           self.cluster.append([])  
           self.cluster_inx = self.cluster_inx + 1  
           self.expandCluster(p_tmp, NeighborPts)     
       
   def expandCluster(self, P, neighbor_points):  
     self.cluster[self.cluster_inx].append(P)  
     iterator = iter(neighbor_points)  
     while True:  
       try:   
         npoint_tmp = iterator.next()  
       except StopIteration:  
         # StopIteration exception is raised after last element  
         break  
       if (not npoint_tmp.visited):  
         #for each point P' in NeighborPts   
         npoint_tmp.visited = True  
         NeighborPts_ = self.regionQuery(npoint_tmp)  
         if (len(NeighborPts_) >= self.MinPts):  
           for j in range(len(NeighborPts_)):  
             neighbor_points.append(NeighborPts_[j])  
       if (not self.checkMembership(npoint_tmp)):  
         #if P' is not yet member of any cluster  
         self.cluster[self.cluster_inx].append(npoint_tmp)  
 #      else:  
 #        print npoint_tmp.show(), 'is belonged to some cluster'  
   
   def checkMembership(self, P):  
     #will return True if point is belonged to some cluster  
     ismember = False  
     for i in range(len(self.cluster)):  
       for j in range(len(self.cluster[i])):  
         if (P.lon == self.cluster[i][j].lon and P.lat == self.cluster[i][j].lat):  
           ismember = True  
     return ismember  
       
   def regionQuery(self, P):  
   #return all points within P's eps-neighborhood, except itself  
     pointInRegion = []  
     for i in range(len(self.DB)):  
       p_tmp = self.DB[i]  
       if (self.dist(P, p_tmp) < self.esp and P.lon != p_tmp.lon and P.lat != p_tmp.lat):  
         pointInRegion.append(p_tmp)  
     return pointInRegion  
   
   def dist(self, p1, p2):  
   #return distance between two point  
     lat1 = p1.lat
     lon1 = p1.lon
     lat2 = p2.lat
     lon2 = p2.lon
     
     radius = 6371 # km

     dlat = math.radians(lat2-lat1)
     dlon = math.radians(lon2-lon1)
     a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
     d = radius * c

     #print d
     return d*100
