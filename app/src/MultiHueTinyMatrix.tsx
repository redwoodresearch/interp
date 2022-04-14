import { ViewComponentProps } from "./proto";
import { toColorRGBA, split_color_wheel, colorSaturation } from "./ui_common";
import { useEffect, useRef } from 'react';

export function MultiHueTinyMatrix(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;
    
    const maxSaturation = 0.5;
    const numColors = vnt.dim_idx_names[0].length;
    const hues = split_color_wheel(numColors);
    const SHAPE = vnt.tensor.shape;
    // so the matrix isn't too small on both dims
    const MIN_MAT_DIM = 150
    const CELLSIZE = Math.max(5, Math.min(Math.floor(MIN_MAT_DIM / SHAPE[1]), Math.floor(MIN_MAT_DIM / SHAPE[2])))

    const canvasRef = useRef(null as null|HTMLCanvasElement)

    useEffect(() => {
        console.log("MULTIHUE RERNDERING",vnt.title)
        const canvas = canvasRef.current
        if (canvas !== null){
            const context = canvas.getContext('2d')
            if (context !== null){
                const array_len = (CELLSIZE * SHAPE[1]) * (CELLSIZE * SHAPE[2]) * 4;
                var tArray = new Uint8ClampedArray(array_len).fill(255);
                for (let n = 0; n < SHAPE[0]; n++) {
                    for (let i = 0; i < SHAPE[1]; i++) {
                        for (let j = 0; j  < SHAPE[2]; j++) {
                            const pick = [n, i, j]
                            let valueInner = 0;
                            if (vnt.tensor !== undefined)
                                valueInner = vnt.tensor.get(...pick);
                            const color = toColorRGBA(valueInner,vnt.colorScale,hues[n])
                            for (let x = 0; x < CELLSIZE; x++) {
                                for (let y = 0; y < CELLSIZE; y++) {
                                    const x_idx = (CELLSIZE * j) + x
                                    const y_idx = (CELLSIZE * i) + y
                                    tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[2]) * 4)] -= 255 - color[0];
                                    tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[2]) * 4) + 1] -= 255 - color[1];
                                    tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[2]) * 4) + 2] -= 255 - color[2];
                                    tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[2]) * 4) + 3] -= 255 - color[3];
                                }
                            }
                        }
                    }
                }

                var imgData = new ImageData(tArray, SHAPE[2] * CELLSIZE)
                context.putImageData(imgData, 0, 0);
                context.drawImage(canvas, 0, 0);
            
            }
        }
    }, [vnt.title]) //currently does not change, but should (whenever the view changes)

    if (vnt.tensor.shape.length !== 3) {
        let err = `MultiHueMatrix only takes 3 dimensions, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }
    else{
        const makeHandler = (setSomething : any) => {
            return (event : any) => {
                const pick = [null, Math.floor(event.nativeEvent.offsetY / CELLSIZE), Math.floor(event.nativeEvent.offsetX / CELLSIZE)];
                setSomething(pick);
            }
        }
    

        return <div>
            <div>
                {vnt.dim_idx_names[0].map((colorName, colorIdx) => {
                    return <span key={colorIdx} 
                                style={{ 
                                    padding: "3px",    
                                    cursor:"pointer",                 
                                    backgroundColor: colorSaturation(hues[colorIdx], maxSaturation)
                                }}
                                onClick={()=>setFocus([colorIdx,null,null])}
                                onMouseEnter={()=>setHover([colorIdx,null,null])}
                            >
                                {colorName}
                            </span>;
                })}
            </div>
            <div onClick={makeHandler(setFocus)} 
                onMouseMove={makeHandler(setHover)}
                style={{
                    position: "relative",
                    border: "1px solid black", 
                    height: (SHAPE[1] * CELLSIZE) + "px",
                    width: (SHAPE[2] * CELLSIZE) + "px", 
                }}
            > 
            <canvas ref={canvasRef} height={SHAPE[1]*CELLSIZE + "px"} width={SHAPE[2]*CELLSIZE + "px"}/> 
            {highlight !== null && highlight[1] !== null && highlight[2] != null &&
                <div style={{
                    top: (highlight[1] * CELLSIZE) + "px",
                    left: (highlight[2] * CELLSIZE) + "px", 
                    width:CELLSIZE + "px", 
                    height:CELLSIZE + "px", 
                    outline:"2px solid black", 
                    zIndex: 1000,
                    position: "absolute", 
                    pointerEvents:"none"}}
                />}
            </div>
        </div>;
    }
}
