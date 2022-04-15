import { ViewComponentProps } from "./proto";


export function Scalar(props: ViewComponentProps) {
    const { vnt } = props;
    if (vnt.tensor.shape.length !== 0) {
        let err = `Scalar only takes 0 dimension, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }

    let value = (!vnt || !vnt.tensor) ? "Loading..." : vnt.tensor.get(0).toFixed(4);
    return (<div>{value}</div>);
}
