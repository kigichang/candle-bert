use candle_core::{Error, Result};
use candle_nn::{Init, LayerNorm};
pub fn layer_norm(size: usize, eps: f64, vb: crate::VarBuilder) -> Result<LayerNorm> {
    // Convert old format to new format if needed from a PyTorch state_dict
    // Safetensors not always in new weight/bias format
    // https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L575
    let weight_tensor_name = ["weight", "gamma"]
        .iter()
        .find(|&name| vb.contains_tensor(name))
        .ok_or_else(|| Error::Msg("Failed to find weight tensor".into()))?;

    let weight = vb.get_with_hints(size, weight_tensor_name, Init::Const(1.))?;

    let bias_tensor_name = ["bias", "beta"]
        .iter()
        .find(|&name| vb.contains_tensor(name))
        .ok_or_else(|| Error::Msg("Failed to find weight tensor".into()))?;

    let bias = vb.get_with_hints(size, bias_tensor_name, Init::Const(0.))?;

    Ok(LayerNorm::new(weight, bias, eps))
}
