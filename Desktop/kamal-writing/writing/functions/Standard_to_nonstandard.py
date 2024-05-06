
import re
import math

 
#a private method
def __local_calculations(x:float, y:float):
    #convertion formula of gcode numbers to kamal numbers
    l1= math.sqrt(math.pow(x,2)+math.pow(950-y,2))*10
    l1Int = int(l1)
    l2 = math.sqrt(math.pow(x-710,2)+math.pow(950-y,2))*10
    l2Int = int(l2) 
    return l1Int,l2Int


def standard_to_locals(output_path:str):
    #traverse in standard gcode and convert it to our grammer
    with open(output_path+"/standardGcode.txt") as file:
        finalFile = open(output_path+"/final.txt", "w")
        finalFile.write("C09,5342,5207,END"+"\n")
        
        while True:
            line = file.readline()
            newLine=""
            if not line: # End of file is reached
                break
            #print(line.strip())
            line = line.strip()
            if(line[:4]=="G1 Z"):#for (G1 Z) order
                pass
            elif(line[:2]=="G1"):
                pattern = 'F....'
                cleared = re.sub(pattern,'',line)
                matchX = re.search(r'X([^\s]+)', cleared)
                matchY = re.search(r'Y([^;]+)', cleared)
                if matchX and matchY:
                    x_value = matchX.group(1)
                    y_value = matchY.group(1)
                    floatX = float(x_value)
                    floatY = float(y_value)
                    finalX, finalY = __local_calculations(floatX,floatY)#these numbers should put into C17 order
                    finalStringX = str(finalX)
                    finalStringY = str(finalY)
                    newLine = "C17,"+finalStringX+","+finalStringY+",2,END" 
                else:
                    raise ValueError("No match found for X or y")
                    
            elif(line[:2]=="M3"):
                newLine = "C13,END"    
            elif(line[:2]=="M5"):
                newLine = "C14,END"
            elif(line[:3] == "G90"):
                pass
            elif(line[:3] == "G91"):
                break 
            else:    
                raise ValueError("invalid input for standard gcode!!!")
            
            if(newLine == ""):
                finalFile.write(newLine)
            else: 
                finalFile.write(newLine+"\n")
            
        file.close()
        finalFile.close()


