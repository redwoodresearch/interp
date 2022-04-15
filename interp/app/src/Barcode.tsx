import { ViewComponentProps } from "./proto";
import { toColorRGBA } from "./ui_common";
import { useEffect, useRef } from 'react';

export function Barcode(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;

    console.log(vnt.tensor.shape, vnt.dim_names);

    const SHAPE = vnt.tensor.shape;
    // so the matrix isn't too small
    const MIN_MAT_DIM = 100;
    const MAX_MAT_DIM = 500;
    let CELLSIZE_Y = 20;
    let CELLSIZE = 5;
    if (SHAPE[0] * CELLSIZE > MAX_MAT_DIM) {
        CELLSIZE = Math.max(1, Math.floor(MAX_MAT_DIM / SHAPE[0]));
    } else if (SHAPE[0] * CELLSIZE < MIN_MAT_DIM) {
        CELLSIZE = Math.floor(MIN_MAT_DIM / SHAPE[0]);
    }
    const canvasRef = useRef(null as null | HTMLCanvasElement);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas !== null) {
            const context = canvas.getContext('2d');
            if (context !== null) {
                // wow this is 100x faster than the old way
                const array_len = (CELLSIZE * SHAPE[0]) * CELLSIZE_Y * 4;
                var tArray = new Uint8ClampedArray(array_len);

                for (let i = 0; i < SHAPE[0]; i++) {
                    const pick = [i];
                    let valueInner = 0;
                    if (vnt.tensor !== undefined)
                        valueInner = vnt.tensor.get(...pick);
                    const color = toColorRGBA(valueInner, vnt.colorScale);
                    for (let x = 0; x < CELLSIZE; x++) {
                        for(let y=0;y<CELLSIZE_Y;y++){
                            
                            tArray[y*4*CELLSIZE*SHAPE[0] + i*CELLSIZE * 4 + x*4] = color[0];
                            tArray[y*4*CELLSIZE*SHAPE[0] + i*CELLSIZE * 4 + x*4 + 1] = color[1];
                            tArray[y*4*CELLSIZE*SHAPE[0] + i*CELLSIZE * 4 + x*4 + 2] = color[2];
                            tArray[y*4*CELLSIZE*SHAPE[0] + i*CELLSIZE * 4 + x*4 + 3] = color[3];
                        }
                    }
                }

                var imgData = new ImageData(tArray, SHAPE[0] * CELLSIZE);
                context.putImageData(imgData, 0, 0);
                context.drawImage(canvas, 0, 0);
            }
        }
    }, [vnt.title]); //currently does not change, but should (whenever the view changes)

    if (vnt.tensor.shape.length !== 1) {
        let err = `Barcode only takes 1 dimensions, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }
    else {
        const makeHandler = (setSomething: any) => {
            return (event: any) => {
                const pick = [Math.floor(event.nativeEvent.offsetY / CELLSIZE),
                Math.floor(event.nativeEvent.offsetX / CELLSIZE)];
                setSomething(pick, event.target, vnt.tensor.get(...pick));
            };
        };

        return <div onClick={makeHandler(setFocus)}
            onMouseMove={makeHandler(setHover)}
            style={{
                position: "relative",
                border: "1px solid black",
                height: CELLSIZE_Y + "px",
                width: (SHAPE[0] * CELLSIZE) + "px"
            }}
        >
            <canvas ref={canvasRef} width={SHAPE[0] * CELLSIZE + "px"} height={CELLSIZE_Y + "px"} {...props} />
            {highlight !== null && highlight[0] !== null && highlight[1] != null &&
                <div
                    style={{
                        top: "0px",
                        left: (highlight[0] * CELLSIZE) + "px",
                        width: CELLSIZE + "px",
                        height: CELLSIZE + "px",
                        outline: "2px solid black",
                        zIndex: 1000,
                        position: "absolute",
                        pointerEvents: "none"
                    }}
                />
            }
        </div>;
    }
}
