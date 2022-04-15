import { toPercentString, meanRest } from "./common";
import { ViewComponentProps } from "./proto";

export function FullFlow(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;
    const c = {
        maxWordWidth: 70,
        fontSize: 12,
        seperation: 60,
        topAxisMargin: 5,
        radius: 7,
        lineWidth: 2,
        color: 'red',
        backgroundOpacityFactor: 0.3,
        maxOpacity: 0.8,
    };
    const toOpacity = (x: number) => {
        return toPercentString(x * c.maxOpacity);
    };
    // to_layer from_layer to from
    const yTop = c.fontSize + 3 + c.topAxisMargin;
    return (<div>
        <svg>
            {vnt.dim_idx_names[2].map((from_name, from_idx) => (<text transform={`translate(${(from_idx + 1.5) * c.maxWordWidth},${c.fontSize})`} text-anchor="middle">{from_name}</text>))}
            <g transform={`translate(0,${yTop})`}>
                {vnt.dim_idx_names[1].map((layer_name, from_layer_idx) => {
                    return (<g transform={`translate(0,${from_layer_idx * c.seperation})`}>
                        <g transform={`translate(${c.maxWordWidth},0)`}>
                            <text transform={`translate(${-c.topAxisMargin - c.radius},${c.fontSize / 2})`} text-anchor="end">{layer_name}</text>
                            {[...vnt.dim_idx_names[3], "OUTPUT"].map((from_name, from_idx) => {
                                let totalOutgoing = 1;
                                if (from_name !== "OUTPUT") {
                                    totalOutgoing = meanRest(vnt.tensor, [null, from_layer_idx, null, from_idx]);
                                }
                                let totalIncoming = 1;
                                if (from_name !== vnt.dim_idx_names[3][0]) {
                                    totalIncoming = meanRest(vnt.tensor, [null, from_layer_idx, from_idx, null]);
                                }
                                if (highlight && highlight.some(x => x !== null) && !(highlight[0] === from_layer_idx && highlight[2] === from_idx)) {
                                    totalOutgoing *= c.backgroundOpacityFactor;
                                }
                                return (
                                    <g transform={`translate(${(from_idx) * c.maxWordWidth + c.radius},0)`}>

                                        {from_name !== "OUTPUT" && vnt.dim_idx_names[1].map((to_name, to_idx) => {

                                            let value = vnt.tensor.get(from_layer_idx, to_idx, from_idx);

                                            if (highlight && highlight.some(x => x !== null) &&
                                                !(highlight[0] === from_layer_idx && highlight[2] === from_idx || (highlight[0] === from_layer_idx + 1 && highlight[1] === to_idx))) {
                                                value *= c.backgroundOpacityFactor;
                                            }

                                            return (<line x1={0} y1={0} y2={c.seperation} x2={(to_idx - from_idx) * c.maxWordWidth} stroke={c.color} strokeWidth={c.lineWidth} opacity={toOpacity(value)}

                                                onMouseEnter={() => setHover && setHover([from_layer_idx, to_idx, from_idx,])}
                                                onClick={() => setFocus && setFocus([from_layer_idx, to_idx, from_idx,])}
                                                onMouseLeave={() => setHover && setHover([null, null, null])}
                                            ></line>);
                                        })}

                                        <circle cx={0} cy={0} r={c.radius} fill={c.color} onMouseEnter={() => setHover && setHover([from_layer_idx, null, from_idx,])} onClick={() => setFocus && setFocus([from_layer_idx, null, from_idx])} onMouseLeave={() => setHover && setHover([null, null, null])}
                                            opacity={toOpacity(totalOutgoing)}></circle>

                                    </g>);
                            })}
                        </g>
                    </g>);
                })}
            </g>
        </svg>
    </div>);
}
