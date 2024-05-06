

import os
from datetime import datetime
from functions import create_SVG,SVG_to_Gcode,Standard_to_nonstandard


def write(path):
    create_SVG.create_SVG(path)
    SVG_to_Gcode.SVG_to_standardGcode(path)
    Standard_to_nonstandard.standard_to_locals(path)


if __name__ == "__main__":
    dt = datetime.today()  
    seconds = dt.timestamp()
    os.mkdir(f"writing/outputs/{seconds}")
    
    write(f"writing/outputs/{seconds}")
     