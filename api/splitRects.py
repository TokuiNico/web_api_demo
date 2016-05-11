"""
    This Module construct set of regions from given geo bounds and size of region
"""
from numpy import deg2rad, rad2deg
from math import sin, cos, atan2, asin

def sphericalOffset(latlng, d, angle):
    """Return the Diagonal geo of region
        INPUT
            latlng: start geo
            d: size of region
            angle: angle
        OUTPUT
            Diagonal geo of region
    """
    
    d = float(d)
    angle = float(angle)
    
    
    if angle == 0:
        lat1 = deg2rad(latlng['lat'])
        lng1 = deg2rad(latlng['lng'])
        lat2 = latlng['lat']
        lng2 = rad2deg( lng1 + atan2( sin(d/6371)*cos(lat1), cos(d/6371)-sin(lat1)*sin(deg2rad(lat2)) ) )
    elif angle == 1:
        lat1 = deg2rad(latlng['lat']);
        lat2 = rad2deg( asin( sin(lat1)*cos(d/6371) - cos(lat1)*sin(d/6371) ) );
        lng2 = latlng['lng'];
    
    return {'lat': lat2, 'lng': lng2}

def splitRects(Sbound, Wbound, Nbound, Ebound, size):
    rectBounds = []
    NW = {'lat': float(Nbound), 'lng': float(Wbound)}
    NS = sphericalOffset(NW, float(size), 0)
    SS = sphericalOffset(NW, float(size), 1)
    
    i = 0
    while 1:
        NE = sphericalOffset(NS, float(i * size), 1)
        SW = sphericalOffset(SS, float(i * size), 1)
        
        a = 0
        while 1:
            rectRecord = {
                'west': SW['lng'],
                'south': SW['lat'],
                'east': NE['lng'],
                'north': NE['lat'],
            }
           

            rectBounds.append(rectRecord)
            SW = sphericalOffset(SW, float(size),0)
            NE = sphericalOffset(NE, float(size),0)
            if SW['lng'] > Ebound: break
            a += 1
        if SW['lat'] < Sbound: break
        i += 1
    
    return rectBounds
    
if __name__ == '__main__':
    Sbound = 41.1;
    Wbound = -8.8;
    Nbound = 41.3;
    Ebound = -8.5;
    size   = 0.3;
    print splitRects(Sbound, Wbound, Nbound, Ebound, size)
    
    