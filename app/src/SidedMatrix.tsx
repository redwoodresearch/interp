import { meanRest, equals } from "./common";
import { PickArgs, ViewComponentProps } from "./proto";
import { toColor, tokenToInlineString } from "./ui_common";

export function SidedMatrix(props: ViewComponentProps) {
    const { vnt, highlight, setHover, setFocus } = props;
    if (vnt.tensor.shape.length !== 2) {
        let err = `SidedMatrix only takes 2 dimensions, got ${vnt.tensor.shape.length}`;
        return <>{err}</>;
    }

    function highlightStyle(pick: PickArgs) {
        return equals(highlight, pick) ? { outline: "3px solid red", outlineOffset: "-3px" } : {};
    }

    return (
        <table style={{ borderCollapse: 'collapse' }}>
            <thead>
                <tr style={{ textAlign: "center" }}><td colSpan={vnt.dim_idx_names[1].length + 2}>{vnt.dim_names[1]}</td></tr>
            </thead>
            <tbody>
                <tr>
                    <td></td><td></td>
                    {vnt.dim_idx_names[1].map((topWord: any, x: number) => {
                        const pick = [null, x];
                        const value = meanRest(vnt.tensor, pick);
                        const handlers = {
                            onClick: (e: any) => setFocus(pick, e.target, value),
                            onMouseEnter: (e: any) => setHover(pick, e.target, value)
                        };
                        return (<td key={x.toString()} style={{
                            writingMode: 'vertical-rl',
                            transform: 'rotate(180deg)',
                            backgroundColor: toColor(value, vnt.colorScale),
                            cursor: "pointer",
                            ...highlightStyle(pick)
                        }} {...handlers} >{tokenToInlineString(topWord)}</td>);
                    }
                    )}
                </tr>
                {vnt.dim_idx_names[0].map((leftWord: any, y: number) => {
                    let pick = [y, null];
                    const value = meanRest(vnt.tensor, pick);
                    let leftHeader = y === 0 ? (
                        <td rowSpan={vnt.dim_idx_names[0].length} style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', textAlign: "center" }}>
                            {vnt.dim_names[0]}
                        </td>) : "";
                    const handlers = {
                        onClick: (e: any) => setFocus(pick, e.target, value),
                        onMouseEnter: (e: any) => setHover(pick, e.target, value)
                    };
                    return (<tr key={y.toString()}>
                        {leftHeader}
                        <td  {...handlers} style={{ backgroundColor: toColor(value, vnt.colorScale), cursor: "pointer", ...highlightStyle(pick) }}>{leftWord}</td>
                        {vnt.dim_idx_names[1].map((topWord: any, x: number) => {
                            const pick = [y, x];
                            let valueInner = 0;
                            if (vnt.tensor !== undefined)
                                valueInner = vnt.tensor.get(...pick);
                            const handlers = {
                                onClick: (e: any) => setFocus(pick, e.target, valueInner),
                                onMouseEnter: (e: any) => setHover(pick, e.target, valueInner)
                            };
                            return <td key={x.toString()} style={{
                                width: 25,
                                height: 25,
                                border: '1px solid black',
                                backgroundColor: toColor(valueInner, vnt.colorScale),
                                cursor: "pointer",
                                ...highlightStyle(pick)
                            }}
                                {...handlers}
                            />;
                        })}
                    </tr>);
                }
                )}
            </tbody>
        </table>
    );
}
