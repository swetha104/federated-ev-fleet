The sample simulation files from the sumo simulator is present here.  Please note that the map_STR.net.xml file is not uploaded here owing to its huge size. Therefore please navigate to this (https://www.openstreetmap.org/#map=6/51.330/10.453) website and extract the osm map.
Please download the osm map as required and use the following commands to generate the files. 
```
netconvert --osm-files map_STR.osm --output-file map_STR.net.xml --geometry.remove --
roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --
tls.join
``` 

```
python utils\randomTrips.py -n map_STR.net.xml --fcd-output -o map_STR.trips.xml -r
map_STR.rou.xml -fcd-output.vType -e 101000 --period 100 --additional-file vehicles.add.xml --
fringe-factor 1 --trip-attributes="type=\"typedist1\"" --verbose
``` 