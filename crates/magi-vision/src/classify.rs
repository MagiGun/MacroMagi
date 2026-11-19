//! Telling pictures apart: a small classifier learned from example pictures of
//! each group (the **AI Vision** block).
//!
//! A picture is shrunk to [`SIDE`]×[`SIDE`] and described by numbers that change
//! little when it moves a few pixels or gets a little brighter: its colors on a
//! coarse grid, its mix of colors, and the directions of its edges. A linear
//! model (softmax regression) learns from those numbers which group a picture
//! is in, and says how sure it is. Every example is learned in several
//! versions – shifted, zoomed, brighter and darker – so a handful of pictures
//! of each group is enough to start with. Learning takes well under a second
//! for a few hundred pictures; nothing is downloaded and no graphics card is needed.

use crate::ImageRef;

/// The side, in pixels, pictures are shrunk to before they are described.
pub const SIDE: usize = 48;
/// The grid the colors are averaged over (a cell is `SIDE / COLOR_GRID` pixels).
const COLOR_GRID: usize = 12;
/// The grid edge directions are counted over, and how many directions.
const EDGE_GRID: usize = 6;
const DIRECTIONS: usize = 8;
/// Levels of red, green and blue in the color mix.
const LEVELS: usize = 4;
/// How many numbers describe a picture.
pub const FEATURES: usize = COLOR_GRID * COLOR_GRID * 3 + EDGE_GRID * EDGE_GRID * DIRECTIONS + LEVELS * LEVELS * LEVELS;

const ITERATIONS: usize = 300;
const LEARNING_RATE: f32 = 0.02;
const WEIGHT_DECAY: f32 = 1e-3;

/// A learned classifier: which group a picture is in, and how sure it is.
#[derive(Clone, Debug)]
pub struct Classifier {
    groups: usize,
    mean: Vec<f32>,
    scale: Vec<f32>,
    /// `groups × FEATURES`, row by row.
    weights: Vec<f32>,
    bias: Vec<f32>,
}

/// How a classifier did on pictures it did not learn from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Checked {
    /// Pictures put in their own group.
    pub right: usize,
    /// Pictures tried.
    pub tried: usize,
}

impl Classifier {
    /// Learns from example pictures, one list per group. `None` when fewer than
    /// two groups have pictures.
    pub fn train(groups: &[Vec<ImageRef<'_>>]) -> Option<Classifier> {
        let samples: Vec<(Vec<f32>, usize)> = groups
            .iter()
            .enumerate()
            .flat_map(|(group, pictures)| pictures.iter().flat_map(move |p| variants(*p).map(move |f| (f, group))))
            .collect();
        Self::fit(&samples, groups.len())
    }

    /// How likely the picture is to be in each group (they add up to 1).
    pub fn probabilities(&self, picture: ImageRef<'_>) -> Vec<f32> {
        let mut x = describe(&shrunk(picture, Crop::WHOLE), 1.0);
        for ((v, m), s) in x.iter_mut().zip(&self.mean).zip(&self.scale) {
            *v = (*v - m) * s;
        }
        softmax(&self.logits(&x))
    }

    /// The likeliest group and how likely it is.
    pub fn classify(&self, picture: ImageRef<'_>) -> (usize, f32) {
        let p = self.probabilities(picture);
        p.iter().copied().enumerate().fold((0, 0.0), |best, (i, v)| if v > best.1 { (i, v) } else { best })
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    fn fit(samples: &[(Vec<f32>, usize)], groups: usize) -> Option<Classifier> {
        let mut counts = vec![0usize; groups];
        for (_, g) in samples {
            counts[*g] += 1;
        }
        if counts.iter().filter(|c| **c > 0).count() < 2 {
            return None;
        }
        // Each number on the same scale, so no kind of description drowns the others.
        let n = samples.len() as f32;
        let mut mean = vec![0f32; FEATURES];
        for (x, _) in samples {
            for (m, v) in mean.iter_mut().zip(x) {
                *m += v / n;
            }
        }
        let mut var = vec![0f32; FEATURES];
        for (x, _) in samples {
            for ((s, v), m) in var.iter_mut().zip(x).zip(&mean) {
                *s += (v - m) * (v - m) / n;
            }
        }
        let scale: Vec<f32> = var.iter().map(|v| 1.0 / (v.sqrt() + 1e-3)).collect();
        let xs: Vec<Vec<f32>> = samples
            .iter()
            .map(|(x, _)| x.iter().zip(&mean).zip(&scale).map(|((v, m), s)| (v - m) * s).collect())
            .collect();
        // A group with few pictures counts as much as one with many.
        let present = counts.iter().filter(|c| **c > 0).count() as f32;
        let weight: Vec<f32> = counts.iter().map(|&c| if c == 0 { 0.0 } else { n / (present * c as f32) }).collect();

        let mut model =
            Classifier { groups, mean, scale, weights: vec![0.0; groups * FEATURES], bias: vec![0.0; groups] };
        // Adam on the mean cross-entropy, with a little weight decay.
        let (mut mw, mut vw) = (vec![0f32; model.weights.len()], vec![0f32; model.weights.len()]);
        let (mut mb, mut vb) = (vec![0f32; groups], vec![0f32; groups]);
        let (beta1, beta2) = (0.9f32, 0.999f32);
        for step in 1..=ITERATIONS {
            let mut gw = vec![0f32; model.weights.len()];
            let mut gb = vec![0f32; groups];
            for (x, (_, g)) in xs.iter().zip(samples) {
                let p = softmax(&model.logits(x));
                for k in 0..groups {
                    let d = (p[k] - if k == *g { 1.0 } else { 0.0 }) * weight[*g] / n;
                    if d == 0.0 {
                        continue;
                    }
                    gb[k] += d;
                    for (w, v) in gw[k * FEATURES..(k + 1) * FEATURES].iter_mut().zip(x) {
                        *w += d * v;
                    }
                }
            }
            for (g, w) in gw.iter_mut().zip(&model.weights) {
                *g += WEIGHT_DECAY * w;
            }
            let (c1, c2) = (1.0 - beta1.powi(step as i32), 1.0 - beta2.powi(step as i32));
            for i in 0..model.weights.len() {
                mw[i] = beta1 * mw[i] + (1.0 - beta1) * gw[i];
                vw[i] = beta2 * vw[i] + (1.0 - beta2) * gw[i] * gw[i];
                model.weights[i] -= LEARNING_RATE * (mw[i] / c1) / ((vw[i] / c2).sqrt() + 1e-8);
            }
            for k in 0..groups {
                mb[k] = beta1 * mb[k] + (1.0 - beta1) * gb[k];
                vb[k] = beta2 * vb[k] + (1.0 - beta2) * gb[k] * gb[k];
                model.bias[k] -= LEARNING_RATE * (mb[k] / c1) / ((vb[k] / c2).sqrt() + 1e-8);
            }
        }
        Some(model)
    }

    fn logits(&self, x: &[f32]) -> Vec<f32> {
        (0..self.groups)
            .map(|k| {
                self.bias[k]
                    + self.weights[k * FEATURES..(k + 1) * FEATURES].iter().zip(x).map(|(w, v)| w * v).sum::<f32>()
            })
            .collect()
    }
}

/// How well the groups can be told apart: each picture is classified by a model
/// learned without it (five rounds, each leaving out every fifth picture).
pub fn check(groups: &[Vec<ImageRef<'_>>]) -> Checked {
    let described: Vec<Vec<(Vec<f32>, usize)>> = groups
        .iter()
        .enumerate()
        .map(|(g, pictures)| pictures.iter().map(|p| (describe(&shrunk(*p, Crop::WHOLE), 1.0), g)).collect())
        .collect();
    let mut checked = Checked::default();
    for round in 0..5 {
        let mut learn = Vec::new();
        let mut test = Vec::new();
        for (g, pictures) in groups.iter().enumerate() {
            for (i, picture) in pictures.iter().enumerate() {
                // A group's only picture is never left out: there would be nothing left to learn it from.
                if i % 5 == round && pictures.len() > 1 {
                    test.push(&described[g][i]);
                } else {
                    learn.extend(variants(*picture).map(|f| (f, g)));
                }
            }
        }
        if test.is_empty() {
            continue;
        }
        let Some(model) = Classifier::fit(&learn, groups.len()) else { continue };
        for (x, g) in test {
            let standardized: Vec<f32> =
                x.iter().zip(&model.mean).zip(&model.scale).map(|((v, m), s)| (v - m) * s).collect();
            let p = softmax(&model.logits(&standardized));
            let best = p.iter().copied().enumerate().fold((0, f32::MIN), |b, (i, v)| if v > b.1 { (i, v) } else { b });
            checked.tried += 1;
            checked.right += usize::from(best.0 == *g);
        }
    }
    checked
}

/// A part of a picture, as fractions of its width and height.
#[derive(Clone, Copy)]
struct Crop {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Crop {
    const WHOLE: Crop = Crop { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };
}

/// The versions of an example that are learned: as it is, shifted a little each
/// way, zoomed in a little, and a little darker and brighter.
fn variants(picture: ImageRef<'_>) -> impl Iterator<Item = Vec<f32>> + '_ {
    const SHIFT: f32 = 0.05;
    let crops = [
        (Crop::WHOLE, 1.0),
        (Crop { x: SHIFT, y: 0.0, w: 1.0 - SHIFT, h: 1.0 }, 1.0),
        (Crop { x: 0.0, y: 0.0, w: 1.0 - SHIFT, h: 1.0 }, 1.0),
        (Crop { x: 0.0, y: SHIFT, w: 1.0, h: 1.0 - SHIFT }, 1.0),
        (Crop { x: 0.0, y: 0.0, w: 1.0, h: 1.0 - SHIFT }, 1.0),
        (Crop { x: SHIFT, y: SHIFT, w: 1.0 - 2.0 * SHIFT, h: 1.0 - 2.0 * SHIFT }, 1.0),
        (Crop::WHOLE, 0.85),
        (Crop::WHOLE, 1.15),
    ];
    crops.into_iter().map(move |(crop, light)| describe(&shrunk(picture, crop), light))
}

/// A part of a picture shrunk to `SIDE`×`SIDE`: each pixel the average of the
/// pixels it covers, as red, green and blue from 0 to 1.
fn shrunk(picture: ImageRef<'_>, crop: Crop) -> Vec<[f32; 3]> {
    let (w, h) = (picture.width.max(1) as f32, picture.height.max(1) as f32);
    let (x0, y0) = (crop.x * w, crop.y * h);
    let (cw, ch) = (crop.w * w, crop.h * h);
    let mut out = vec![[0f32; 3]; SIDE * SIDE];
    for (row, line) in out.chunks_mut(SIDE).enumerate() {
        let top = (y0 + ch * row as f32 / SIDE as f32) as u32;
        let bottom = ((y0 + ch * (row + 1) as f32 / SIDE as f32) as u32).max(top + 1).min(picture.height);
        for (col, cell) in line.iter_mut().enumerate() {
            let left = (x0 + cw * col as f32 / SIDE as f32) as u32;
            let right = ((x0 + cw * (col + 1) as f32 / SIDE as f32) as u32).max(left + 1).min(picture.width);
            let (mut sum, mut count) = ([0u32; 3], 0u32);
            for y in top.min(picture.height.saturating_sub(1))..bottom {
                let start = (y * picture.width + left.min(picture.width.saturating_sub(1))) as usize * 4;
                let end = (y * picture.width + right) as usize * 4;
                for px in picture.rgba[start..end].as_chunks::<4>().0 {
                    sum[0] += u32::from(px[0]);
                    sum[1] += u32::from(px[1]);
                    sum[2] += u32::from(px[2]);
                    count += 1;
                }
            }
            if count > 0 {
                *cell = sum.map(|s| s as f32 / (count as f32 * 255.0));
            }
        }
    }
    out
}

/// The numbers a shrunk picture is told apart by (`light` makes it darker or brighter first).
fn describe(pixels: &[[f32; 3]], light: f32) -> Vec<f32> {
    let pixel = |x: usize, y: usize| pixels[y * SIDE + x].map(|v| (v * light).min(1.0));
    let mut out = Vec::with_capacity(FEATURES);
    // Colors on a coarse grid.
    let cell = SIDE / COLOR_GRID;
    for gy in 0..COLOR_GRID {
        for gx in 0..COLOR_GRID {
            let mut sum = [0f32; 3];
            for y in gy * cell..(gy + 1) * cell {
                for x in gx * cell..(gx + 1) * cell {
                    let p = pixel(x, y);
                    for c in 0..3 {
                        sum[c] += p[c];
                    }
                }
            }
            out.extend(sum.map(|s| s / (cell * cell) as f32));
        }
    }
    // The directions of edges, counted per cell and weighted by their strength.
    let gray = |x: usize, y: usize| {
        let p = pixel(x, y);
        0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
    };
    let mut edges = vec![0f32; EDGE_GRID * EDGE_GRID * DIRECTIONS];
    let edge_cell = SIDE / EDGE_GRID;
    for y in 1..SIDE - 1 {
        for x in 1..SIDE - 1 {
            let gx = gray(x + 1, y) - gray(x - 1, y);
            let gy = gray(x, y + 1) - gray(x, y - 1);
            let strength = (gx * gx + gy * gy).sqrt();
            if strength < 1e-4 {
                continue;
            }
            // Direction without its sign (0..π): a dark-to-light edge and a light-to-dark one match.
            let angle = gy.atan2(gx).rem_euclid(std::f32::consts::PI);
            let bin = ((angle / std::f32::consts::PI * DIRECTIONS as f32) as usize).min(DIRECTIONS - 1);
            let at = ((y / edge_cell) * EDGE_GRID + x / edge_cell) * DIRECTIONS + bin;
            edges[at] += strength;
        }
    }
    let total = edges.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    out.extend(edges.iter().map(|v| v / total));
    // The mix of colors, wherever they are.
    let mut mix = vec![0f32; LEVELS * LEVELS * LEVELS];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let level = |v: f32| ((v * LEVELS as f32) as usize).min(LEVELS - 1);
            let p = pixel(x, y);
            mix[(level(p[0]) * LEVELS + level(p[1])) * LEVELS + level(p[2])] += 1.0 / (SIDE * SIDE) as f32;
        }
    }
    out.extend(mix);
    out
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let top = logits.iter().copied().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bitmap;

    /// A picture of a solid color with a bar across it, moved a little and lit differently.
    fn picture(color: [u8; 3], bar_vertical: bool, shift: u32, light: f32) -> Bitmap {
        let (w, h) = (64u32, 40u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let on_bar = if bar_vertical { (x + 64 - shift) % 64 / 8 == 3 } else { (y + 40 - shift) % 40 / 6 == 3 };
                let c = if on_bar { [250, 250, 250] } else { color };
                rgba.extend(c.map(|v| (f32::from(v) * light).min(255.0) as u8));
                rgba.push(255);
            }
        }
        Bitmap { width: w, height: h, rgba }
    }

    #[test]
    fn it_tells_groups_apart_and_is_sure_of_them() {
        let menu: Vec<Bitmap> = (0..4).map(|i| picture([30, 60, 160], true, i, 1.0)).collect();
        let game: Vec<Bitmap> = (0..4).map(|i| picture([40, 140, 50], false, i, 1.0)).collect();
        let groups = vec![menu.iter().map(Bitmap::as_ref).collect(), game.iter().map(Bitmap::as_ref).collect()];
        let model = Classifier::train(&groups).expect("two groups");
        // Unseen versions: moved further and darker than any example.
        let (group, sure) = model.classify(picture([30, 60, 160], true, 6, 0.8).as_ref());
        assert_eq!(group, 0);
        assert!(sure > 0.8, "{sure}");
        let (group, sure) = model.classify(picture([40, 140, 50], false, 5, 1.1).as_ref());
        assert_eq!(group, 1);
        assert!(sure > 0.8, "{sure}");
        let p = model.probabilities(picture([40, 140, 50], false, 0, 1.0).as_ref());
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert_eq!(check(&groups), Checked { right: 8, tried: 8 });
    }

    #[test]
    fn it_needs_two_groups_with_pictures() {
        let one = picture([10, 10, 10], true, 0, 1.0);
        assert!(Classifier::train(&[vec![one.as_ref()], vec![]]).is_none());
        assert!(Classifier::train(&[vec![one.as_ref()]]).is_none());
    }

    #[test]
    fn odd_sizes_are_described_too() {
        for (w, h) in [(1u32, 1u32), (3, 200), (500, 2), (47, 49)] {
            let bitmap = Bitmap { width: w, height: h, rgba: vec![128; (w * h * 4) as usize] };
            let x = describe(&shrunk(bitmap.as_ref(), Crop::WHOLE), 1.0);
            assert_eq!(x.len(), FEATURES);
            assert!(x.iter().all(|v| v.is_finite()));
        }
    }
}
