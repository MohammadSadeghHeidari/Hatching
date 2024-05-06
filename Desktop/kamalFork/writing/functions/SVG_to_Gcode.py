
from svg_to_gcode.svg_parser import parse_file
from svg_to_gcode.compiler import Compiler, interfaces


def SVG_to_standardGcode(output_path:str):
    # Instantiate a compiler with specific parameters
    gcode_compiler = Compiler(interfaces.Gcode, movement_speed=900, cutting_speed=300, pass_depth=5)

    # Parse an SVG file into geometric curves
    curves = parse_file(output_path+"/output.svg")

    # Append the curves to the compiler
    gcode_compiler.append_curves(curves)

    # Compile the curves to G-code and save to a file
    gcode_compiler.compile_to_file(output_path+"/standardGcode.txt", passes=2)