## High Priority

## Medium priority
- State serialization
- Make widgets nicer
    - Gain on colormaps
    - Colorbars
    - Icon/emoji
    - Loading indicator
- Useful widget for high-dimensional tensors
    - Show summary statistics
    - Smarter suggestions for reasonable things to drill into
    

# Low priority:
- Re-add amax reduction unless there was some good reason it was previously removed. 
- Should we allow doing graphs where the dim_types don't match the requirements? Leads to meaningless plots.
- What are we missing in the interface?
    - Transpose or flatten axes
- "Clear focus" button should show what you're clearing
- Rename things (mostly types) for clarity
    - Focus class is kinda not worth it versus just common functions of PickArgs
- Design right clicking support
- Clean up "any" types in Typescript
- Clean up todos and commented code
- Clean up and read https://docs.google.com/document/d/1NvoZOgVsszvaBwjF88flFjPUpg5YN9HXlEH-TDf3HRQ/edit
- Think about fetch vs websocket
    - Needs benchmarking - interesting blog post? I'm assuming websocket is better for hover requests
- Pagination support
    - ScrollViews, panning, filter operations like slider for K in top-K
- Look at doing cached client-side reductions to reduce number of network calls - probably faster? 


[Notes from Sidney discussion](https://docs.google.com/document/d/1OC2J8ZNwi8FrjsPHjfNXJH3jBir2zzzLpKbFrNqRa2o/edit#)
