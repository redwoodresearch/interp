import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { range } from "./common";
import { css_prop_by_id } from "./ui_common";
import { ViewComponentProps } from "./proto";

export default function LinePlot(props: ViewComponentProps) {
    const { setFocus, setHover, vnt } = props;
    const series = [];
    if (vnt.tensor.shape.length === 2) {
        for (let sIdx = 0; sIdx < vnt.tensor.shape[0]; sIdx++) {
            series.push({ data: range(vnt.tensor.shape[1]).map((x, i) => vnt.tensor.get(sIdx, i)), name: vnt.dim_idx_names[0][sIdx] });
        }
    } else if (vnt.tensor.shape.length === 1) {
        series.push({ data: range(vnt.tensor.shape[0]).map((x, i) => vnt.tensor.get(i)), });
    } else {
        let err = `Lineplot only takes 1 or 2 dimensions, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }
    const options = {
        series,
        title: { text: vnt.title },
        xAxis: { type: "category", categories: vnt.dim_idx_names[vnt.tensor.shape.length === 2 ? 1 : 0] },
        yAxis: { title: { text: vnt.units } },
        plotOptions: {
            series: {
                allowPointSelect: true,
                animation:false,
                point: {
                    events: {
                        select: (event: any) => {
                            if (vnt.dim_names.length === 1) {
                                setFocus([event.target.index]);
                            } else {
                                setFocus([event.target.series.index, event.target.index]);

                            }
                        },
                        mouseOver: (event: any) => {
                            if (vnt.dim_names.length === 1) {
                                setHover([event.target.index]);
                            } else {
                                setHover([event.target.series.index, event.target.index]);
                            }
                        },
                        mouseOut: (event: any) => {
                            if (vnt.dim_names.length === 1) {
                                setHover([null]);
                            } else {
                                setHover([null, null]);
                            }
                        }
                    }
                }
            }
        },
        tooltip: {
            pointFormat: "{series.name}: <b>{point.y:.4f} " + vnt.units + "</b></br>",
            shared: true
        },
        chart: {
            animation:false,
            style : {
                fontFamily: css_prop_by_id("line_plot", "font-family")
            }
        }
    };
    return <div className="line_plot" style={{width:"100%",minWidth:"350px"}}>
    <HighchartsReact
        highcharts={Highcharts}
        options={options}
    /></div>;
}
