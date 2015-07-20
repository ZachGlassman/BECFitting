#pragma rtGlobals=1		// Use modern global access method.

//procedure for writing matrices to file for processing
Function exportImages(starti, endi, pathStr,aname)
	variable starti
	variable endi
	string pathStr
	string aname
	variable i 
	string name
	string pathFrom
	string pathTo
	NewPath/O pathTo pathStr
	for(i = starti; i < endi + 1; i+=1)
	       if (i<1000)
	       	if (i<100)
	       		if (i<10)
	       			name = "matrix" + "000"  +num2str(i) + "_0"
	       		else
	       		      name = "matrix" + "00"  +num2str(i) + "_0"
	       		endif
	       	else
			    name = "matrix" + "0"  +num2str(i) + "_0"
			endif
		else
			name = "matrix"  +num2str(i) + "_0"
		endif
		print name
		pathFrom = ":Matrices:" + name
		Save/G/O/M = "\r\n"/U={1,0,0,0}/P = pathTo $pathFrom as (aname + name + ".txt")
	endfor
	
end

//Example pathTo:
//  "C:Users:sodiumpa:Documents:IGOR Experiments:BEC mixture:2014_06_25:images"

//Consider for pathFrom:
//  root:Matrices:matrix0263_0

