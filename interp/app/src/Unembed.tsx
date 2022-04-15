import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { range } from "./common";
import { css_prop_by_id, ShowQueriedLogits } from "./ui_common";
import { QueriedLogits, ViewComponentProps } from "./proto";
import { useEffect, useState } from "react";
import { Tops } from "./Tops";

export default function Unembed(props: ViewComponentProps) {
    const [unembedded,setUnembedded] = useState(null as null|QueriedLogits)
    useEffect(()=>{
        if(props.unembedder){
            props.unembedder(range(props.vnt.dim_idx_names[0].length).map(i=> props.vnt.tensor.get(i))).then((ql:QueriedLogits)=>setUnembedded(ql))
        }
    },[props.vnt.title])
    if (!props.unembedder){
        return (<p>Need unembedder option from backend</p>)
    }
    if (!unembedded){
        return <p>Loading</p>
    }
    return (<ShowQueriedLogits ql={unembedded}/>)
}