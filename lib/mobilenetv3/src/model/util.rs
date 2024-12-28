// From Pytorch/torchvision, which they in turn used from TF:
// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
pub(crate) fn make_divisible(
    v: f64,
    divisor: usize,
    min_value: Option<usize>,
) -> usize {
    let min_value = min_value.unwrap_or(divisor);
    let new_v =
        (v + (divisor as f64 / 2.0)).floor() as usize / divisor * divisor;
    let new_v = new_v.max(min_value);

    if (new_v as f64) < (0.9 * v) {
        return new_v + divisor;
    } else {
        return new_v;
    };
}

pub(crate) fn adjust_channels(channels: usize, width_mult: f64) -> usize {
    return make_divisible(channels as f64 * width_mult, 8, None);
}
