
import cairo
import arabic_reshaper



def create_SVG(output_path:str):
    
    # Creating a SVG surface
    with cairo.SVGSurface(output_path+"/output.svg", 900, 900) as surface:
        # Creating a cairo context object for SVG surface
        context = cairo.Context(surface)
    
        # Setting color of the context
        context.set_source_rgb(0, 0, 0) # Black color
    
        # Approximate text height
        context.set_font_size(100)
    
        # Font Style
        context.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    
        # Position for the text
        context.move_to(50, 400)

        # Displays the text
        #context.show_text("Hello SVG!")
        print("1)persian:")
        print("2)english:")
        language = input("choose one of them: ")
        if language=="1":
            text = input("please write the persian text:"+"\n")
            reshaped_text = arabic_reshaper.reshape(text)
            context.text_path(reshaped_text[::-1])
        elif language=="2":
            text=input("please write the english text:"+ "\n")
            context.text_path(text)
        else:
            raise ValueError("invalid number for languages")
        
        
        # Set the width of the outline
        context.set_line_width(1)
    
        # Stroke the text to make it hollow
        context.stroke()
        #print("File Saved")
    
