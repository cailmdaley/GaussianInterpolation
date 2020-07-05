export Static, InPlace, ImFilter
export interp

function interp(data, model, design)
    for i in eachindex(data)
        is = design_is(data, model, design, i)
        predict!(data, model, i, is)
    end
    data.ds_predict
end
