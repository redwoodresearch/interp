import { ViewComponentProps } from "./proto";
import { toColorRGBA } from "./ui_common";
import { useEffect, useRef } from 'react';

export function TinyMatrix(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;

    console.log(vnt.tensor.shape, vnt.dim_names)

    const SHAPE = vnt.tensor.shape
    // so the matrix isn't too small
    const MIN_MAT_DIM = 150
    const MAX_MAT_DIM=1000
    let CELLSIZE=5
    if (Math.max(...SHAPE)*CELLSIZE>MAX_MAT_DIM){
        CELLSIZE=Math.max(1,Math.floor(MAX_MAT_DIM/Math.max(...SHAPE)))
    }else if (Math.min(...SHAPE)*CELLSIZE<MIN_MAT_DIM){
        CELLSIZE=Math.floor(MIN_MAT_DIM/Math.min(...SHAPE))
    }
    const canvasRef = useRef(null as null|HTMLCanvasElement)

    useEffect(() => {
        const canvas = canvasRef.current
        if (canvas !== null){
            const context = canvas.getContext('2d')
            if (context !== null){
                // wow this is 100x faster than the old way
                const array_len = (CELLSIZE * SHAPE[0]) * (CELLSIZE * SHAPE[1]) * 4;
                var tArray = new Uint8ClampedArray(array_len)
                
                for (let i = 0; i < SHAPE[0]; i++) {
                    for (let j = 0; j  < SHAPE[1]; j++) {
                        const pick = [i, j]
                        let valueInner = 0;
                        if (vnt.tensor !== undefined)
                            valueInner = vnt.tensor.get(...pick);
                        const color = toColorRGBA(valueInner,vnt.colorScale)
                        for (let x = 0; x < CELLSIZE; x++) {
                            for (let y = 0; y < CELLSIZE; y++) {
                                const x_idx = (CELLSIZE * j) + x
                                const y_idx = (CELLSIZE * i) + y
                                tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[1]) * 4)] = color[0];
                                tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[1]) * 4) + 1] = color[1];
                                tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[1]) * 4) + 2] = color[2];
                                tArray[(x_idx * 4) + (y_idx * (CELLSIZE * SHAPE[1]) * 4) + 3] = color[3];
                            }
                        }
                    }
                }

                var imgData = new ImageData(tArray, SHAPE[1] * CELLSIZE)
                context.putImageData(imgData, 0, 0);
                context.drawImage(canvas, 0, 0);
            }
        }
    }, [vnt.title]) //currently does not change, but should (whenever the view changes)

    if (vnt.tensor.shape.length !== 2) {
        let err = `TinyMatrix only takes 2 dimensions, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }
    else{
        const makeHandler = (setSomething : any) => {
            return (event : any) => {
                const pick = [Math.floor(event.nativeEvent.offsetY / CELLSIZE), 
                                Math.floor(event.nativeEvent.offsetX / CELLSIZE)];
                setSomething(pick, event.target, vnt.tensor.get(...pick));
            }
        }

        return <div onClick={makeHandler(setFocus)} 
            onMouseMove={makeHandler(setHover)} 
            style={{
                position: "relative",
                border: "1px solid black", 
                height: (SHAPE[0] * CELLSIZE) + "px",
                width: (SHAPE[1] * CELLSIZE) + "px"
            }}
        > 
            <canvas ref={canvasRef} height={SHAPE[0]*CELLSIZE + "px"} width={SHAPE[1]*CELLSIZE + "px"} {...props}/> 
            {highlight !== null && highlight[0] !== null && highlight[1] != null && 
                <div 
                    style={{
                        top: (highlight[0] * CELLSIZE) + "px",
                        left: (highlight[1] * CELLSIZE) + "px", 
                        width:CELLSIZE + "px", 
                        height:CELLSIZE + "px", 
                        outline:"2px solid black", 
                        zIndex: 1000,
                        position: "absolute", 
                        pointerEvents:"none"
                    }}
                />
            }
        </div>
    }
}
