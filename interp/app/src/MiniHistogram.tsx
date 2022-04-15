import { ViewComponentProps } from "./proto";
import { inf, sup } from "ndarray-ops";
import { ndarrayWalk, range } from "./common";


export function MiniHistogram(props: ViewComponentProps) {
    const { vnt } = props;
    const numBuckets = 30;
    const pixelWidth = 120;
    const min = inf(vnt.tensor);
    const max = sup(vnt.tensor);
    const spread = max - min;
    const bucketMins = range(numBuckets).map(i => min + spread * i / numBuckets);
    const bucketCounts = range(numBuckets).map(x => 0);
    ndarrayWalk(vnt.tensor, (idx: number[], val: number) => {
        if (!isNaN(val)) {
            const bucket = Math.min(numBuckets - 1, Math.floor((val - min) / spread * numBuckets));
            bucketCounts[bucket] += 1;
        }
    });
    const maxBucketCount = Math.max(...bucketCounts);
    return (<div style={{ display: "flex", flexDirection: "row", fontSize: "14px" }}><div style={{ display: "flex", flexDirection: "column" }}>
        {range(numBuckets).map(i => <div key={i} style={{ display: "flex", flexDirection: "row", padding: "0 4px 0 0" }}>
            <div>{bucketMins[i].toFixed(4)}</div></div>)}</div>
        <div style={{ display: "flex", flexDirection: "column" }}>
            {range(numBuckets).map(i =>
                <div key={i} style={{ width: `${Math.floor(bucketCounts[i] / maxBucketCount * pixelWidth)}px`, backgroundColor: "#CCCCFF" }}>{bucketCounts[i]}</div>)}
        </div>
    </div>);
}
