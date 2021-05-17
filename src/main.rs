#![feature(unboxed_closures)]
#![feature(fn_traits)]

extern crate anyhow;
extern crate rustfft;
extern crate rusttype;
extern crate splines;

use ::core::f64;
use anyhow::anyhow;
use std::{
    f64::consts::PI,
    fmt::Write,
    ops::{Div, Mul, RangeInclusive},
};

struct FourierSeries {
    coeffs: Vec<(f64, f64)>,
    period: f64,
}

enum FourierCoefficients<
    'a,
    T: Iterator<Item = f64> = Box<dyn Iterator<Item = f64>>,
    F: Fn(f64) -> f64 = Box<dyn Fn(f64) -> f64>,
    Fi: Fn(i32) -> (f64, f64) = Box<dyn Fn(i32) -> (f64, f64)>,
> {
    Unzipped(T, T),
    Zipped(Vec<(f64, f64)>),
    ApproxFn(F, i32, &'a RangeInclusive<f64>),
    /// Iterates through provided range, applying the function Fi to them, which'll produce the coefficients.
    IterFn(Fi, RangeInclusive<i32>),
    /// data has to be sorted.
    ApproxDataLinearSpline(Vec<f64>),
}

impl<'a, T: Iterator<Item = f64>, F: Fn(f64) -> f64, Fi: Fn(i32) -> (f64, f64)>
    FourierCoefficients<'a, T, F, Fi>
{
    fn get_coeffs(self) -> Vec<(f64, f64)> {
        use rustfft::num_complex::Complex;
        match self {
            Self::Unzipped(iterator_a_coeffs, iterator_b_coeffs) => iterator_a_coeffs
                .into_iter()
                .zip(iterator_b_coeffs.into_iter())
                .collect(),
            Self::Zipped(coeffs) => coeffs,
            Self::IterFn(f, range) => {
                let mut result = Vec::with_capacity(*range.end() as usize);
                for n in range {
                    result.push(f(n))
                }
                result
            }
            Self::ApproxFn(f, n, range) => {
                let n = 2 * n;

                let mut data = Vec::with_capacity(n as usize);
                let mut fft_planner = rustfft::FftPlanner::new();
                let fft = fft_planner.plan_fft_forward(n as usize);

                let start = range.start();
                let end = range.end();
                let delta_time = (end - start) / (n as f64);

                let mut time = *start;
                let mut t = || {
                    let prev = time;
                    time += delta_time;
                    prev
                };

                for _ in 0..n {
                    let real_val = f(t());
                    data.push(Complex::new(real_val, 0e0));
                }

                fft.process(&mut data);

                let ndiv2 = n / 2;
                let mut result = Vec::with_capacity(ndiv2 as usize);

                for coeff in &data[0..(ndiv2 as usize)] {
                    result.push((2e0 * (coeff.re) / n as f64, -2e0 * (coeff.im) / n as f64));
                }

                result[0].0 /= 2e0;
                result
            }
            _ => {
                panic!("not implemented yet!")
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TimestampsIterator {
    t: f64,
    dt: f64,
    time_stop: f64,
}

impl Iterator for TimestampsIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.t >= self.time_stop {
            None
        } else {
            self.t += self.dt;
            Some(self.t)
        }
    }
}

impl FourierSeries {
    fn new(fourier_coefficients: FourierCoefficients, period: f64) -> Self {
        Self {
            coeffs: fourier_coefficients.get_coeffs(),
            period: period,
        }
    }
    fn period(&self) -> f64 {
        self.period
    }

    fn sample(&self, t: f64) -> f64 {
        let mut result = 0e0;
        let period = self.period();
        for (n, coeff) in self.coeffs.iter().enumerate() {
            let n = n as f64;
            let a = coeff.0;
            let b = coeff.1;
            result += a * (2e0 * PI / period * n * t).cos() + b * (2e0 * PI / period * n * t).sin();
        }
        result
    }

    fn timestamps(&self, delta_time: f64) -> TimestampsIterator {
        TimestampsIterator {
            t: 0e0,
            dt: delta_time,
            time_stop: self.period(),
        }
    }

    fn equation(&self) -> anyhow::Result<String> {
        let mut f = String::new();
        for n in 0..self.coeffs.len() {
            let coeff = self.coeffs[n];
            let a = coeff.0.mul(1000e0).round().div(1000e0);
            let b = coeff.1.mul(1000e0).round().div(1000e0);
            let c = (2e0 * PI / self.period() * n as f64)
                .mul(1000e0)
                .round()
                .div(1000e0);
            if n == 0 {
                if c != 0e0 {
                    if a.abs() > 0.010 {
                        f.write_fmt(format_args!("{}cos({}*t) ", a, c))?;
                    }

                    if b != 0e0 && b.abs() > 0.010 {
                        if b.is_sign_negative() {
                            f.write_fmt(format_args!("- {}sin({}*t) ", b.abs(), c))?;
                        } else {
                            f.write_fmt(format_args!("+ {}sin({}*t) ", b, c))?;
                        }
                    }
                } else {
                    f.write_fmt(format_args!("{} ", a))?;
                }
            } else {
                if c != 0e0 {
                    if a.abs() > 0.010 {
                        if a.is_sign_negative() {
                            f.write_fmt(format_args!("- {}cos({}*t) ", a.abs(), c))?;
                        } else {
                            f.write_fmt(format_args!("+ {}cos({}*t) ", a, c))?;
                        }
                    }

                    if b != 0e0 && b.abs() > 0.010 {
                        if b.is_sign_negative() {
                            f.write_fmt(format_args!("- {}sin({}*t) ", b.abs(), c))?;
                        } else {
                            f.write_fmt(format_args!("+ {}sin({}*t) ", b, c))?;
                        }
                    }
                } else {
                    if a.is_sign_negative() {
                        f.write_fmt(format_args!("- {} ", a.abs()))?;
                    } else {
                        f.write_fmt(format_args!("+ {} ", a))?;
                    }
                }
            }
        }

        Ok(f)
    }
}

impl FnOnce<(f64,)> for FourierSeries {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        self.sample(args.0)
    }
}

impl FnMut<(f64,)> for FourierSeries {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        self.sample(args.0)
    }
}

impl Fn<(f64,)> for FourierSeries {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        self.sample(args.0)
    }
}

struct Scale1d;
impl Scale1d {
    fn scale<'a>(points: &mut Vec<f64>, bounds: Option<(f64, f64)>) {
        let min = *points
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let max = *points
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();

        let bounds = bounds.unwrap_or((-1e0, 1e0));
        let delta_bounds = bounds.1 - bounds.0;
        let delta_points = max - min;

        for p in points.iter_mut() {
            *p = bounds.0 + (*p - min) * (delta_bounds / delta_points);
        }
    }
}

trait ScalingForVec2d {
    fn scale(&self, points: &mut Vec<Vec2d>, bounds: Option<((f64, f64), (f64, f64))>);
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Vec2d {
    x: f64,
    y: f64,
}

impl std::ops::Mul<f64> for Vec2d {
    type Output = Vec2d;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Div<f64> for Vec2d {
    type Output = Vec2d;

    fn div(self, rhs: f64) -> Self::Output {
        Self::Output {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl std::ops::Mul<Vec2d> for f64 {
    type Output = Vec2d;

    fn mul(self, rhs: Vec2d) -> Self::Output {
        Self::Output {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl std::ops::Div<Vec2d> for f64 {
    type Output = Vec2d;

    fn div(self, rhs: Vec2d) -> Self::Output {
        Self::Output {
            x: self / rhs.x,
            y: self / rhs.y,
        }
    }
}

impl std::ops::Add<Vec2d> for Vec2d {
    type Output = Vec2d;

    fn add(self, rhs: Vec2d) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub<Vec2d> for Vec2d {
    type Output = Vec2d;

    fn sub(self, rhs: Vec2d) -> Self::Output {
        Self::Output {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

fn vec2(x: impl Into<f64>, y: impl Into<f64>) -> Vec2d {
    Vec2d {
        x: x.into(),
        y: y.into(),
    }
}

struct ScaleVec2d;
impl ScaleVec2d {
    fn scale(points: &mut Vec<Vec2d>, bounds: Option<((f64, f64), (f64, f64))>) {
        let (mut minx, mut miny, mut maxx, mut maxy) =
            (std::f64::MAX, std::f64::MAX, std::f64::MIN, std::f64::MIN);

        for p in points.iter() {
            if p.x < minx {
                minx = p.x;
            }

            if p.y < miny {
                miny = p.y;
            }

            if p.x > maxx {
                maxx = p.x;
            }

            if p.y > maxy {
                maxy = p.y;
            }
        }

        let bounds = bounds.unwrap_or(((-1e0, 1e0), (-1e0, 1e0)));
        let delta_bounds_x = bounds.0 .1 - bounds.0 .0;
        let delta_bounds_y = bounds.1 .1 - bounds.1 .0;
        let delta_points_x = maxx - minx;
        let delta_points_y = maxy - miny;

        for p in points.iter_mut() {
            p.x = bounds.0 .0 + (p.x - minx) * (delta_bounds_x / delta_points_x);
            p.y = bounds.1 .0 + (p.y - miny) * (delta_bounds_y / delta_points_y);
        }
    }
}

impl splines::Interpolate<f64> for Vec2d {
    fn step(t: f64, threshold: f64, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f64, a: Self, b: Self) -> Self {
        a * (1. - t) + b * t
    }

    fn cosine(t: f64, a: Self, b: Self) -> Self {
        let cos_nt = (1. - (t * std::f64::consts::PI).cos()) * 0.5;
        Self::lerp(cos_nt, a, b)
    }

    fn cubic_hermite(
        t: f64,
        x: (f64, Self),
        a: (f64, Self),
        b: (f64, Self),
        y: (f64, Self),
    ) -> Self {
        let two_t = t * 2.;
        let three_t = t * 3.;
        let t2 = t * t;
        let t3 = t2 * t;
        let two_t3 = t3 * two_t;
        let three_t2 = t2 * three_t;

        let m0 = (b.1 - x.1) / (b.0 - x.0);
        let m1 = (y.1 - a.1) / (y.0 - a.0);

        a.1 * (two_t3 - three_t2 + 1.)
            + m0 * (t3 - t2 * two_t + t)
            + b.1 * (three_t2 - two_t3)
            + m1 * (t3 - t2)
    }

    fn quadratic_bezier(t: f64, a: Self, u: Self, b: Self) -> Self {
        let one_t = 1. - t;
        let one_t2 = one_t * one_t;

        u + (a - u) * one_t2 + (b - u) * t * t
    }

    fn cubic_bezier(t: f64, a: Self, u: Self, v: Self, b: Self) -> Self {
        let one_t = 1. - t;
        let one_t2 = one_t * one_t;
        let one_t3 = one_t2 * one_t;
        let t2 = t * t;

        a * one_t3 + (u * one_t2 * t + v * one_t * t2) * 3. + b * t2 * t
    }

    fn cubic_bezier_mirrored(t: f64, a: Self, u: Self, v: Self, b: Self) -> Self {
        Self::cubic_bezier(t, a, u, b + b - v, b)
    }
}

#[derive(Debug)]
struct PointCollector {
    collected_points: Vec<Vec2d>,
    prev: Vec2d,
    delta_time: f64,
}

impl PointCollector {
    fn new(delta_time: f64) -> Self {
        Self {
            collected_points: vec![],
            prev: Vec2d::default(),
            delta_time: delta_time,
        }
    }

    fn quad(p0: Vec2d, p1: Vec2d, p2: Vec2d, t: f64) -> Vec2d {
        (1e0 - t).powi(2) * p0 + 2e0 * (1e0 - t) * t * p1 + t.powi(2) * p2
    }

    fn cubic(p0: Vec2d, p1: Vec2d, p2: Vec2d, p3: Vec2d, t: f64) -> Vec2d {
        (1e0 - t).powi(3) * p0
            + 3e0 * (1e0 - t).powi(2) * t * p1
            + 3e0 * (1e0 - t) * t.powi(2) * p2
            + t.powi(3) * p3
    }

    fn line(p0: Vec2d, p1: Vec2d, t: f64) -> Vec2d {
        p0 + (p1 - p0) * t
    }
}

impl rusttype::OutlineBuilder for PointCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.prev = vec2(x, y)
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = vec2(x, y);
        let mut t = 0e0;
        while t <= 1e0 {
            self.collected_points.push(Self::line(self.prev, p, t));
            t += self.delta_time;
        }

        self.prev = p;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p = vec2(x, y);
        let mut t = 0e0;
        while t <= 1e0 {
            self.collected_points
                .push(Self::quad(self.prev, vec2(x1, y1), p, t));
            t += self.delta_time;
        }

        self.prev = p;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p = vec2(x, y);
        let mut t = 0e0;
        while t <= 1e0 {
            self.collected_points
                .push(Self::cubic(self.prev, vec2(x1, y1), vec2(x2, y2), p, t));
            t += self.delta_time;
        }

        self.prev = p;
    }

    fn close(&mut self) {}
}

pub fn char_to_glyph<'a>(font: &'a rusttype::Font, c: char) -> rusttype::Glyph<'a> {
    font.layout(
        &c.to_string(),
        rusttype::Scale::uniform(1e0),
        rusttype::point(0e0, 0e0),
    )
    .nth(0)
    .unwrap()
    .into_unpositioned()
    .into_unscaled()
}

pub fn text_to_glyphs<'a>(font: &'a rusttype::Font, text: &str) -> Vec<rusttype::Glyph<'a>> {
    font.layout(
        text,
        rusttype::Scale::uniform(1e0),
        rusttype::point(0e0, 0e0),
    )
    .map(|s_p_glyph| s_p_glyph.into_unpositioned().into_unscaled())
    .collect()
}

pub fn glyph_to_points(g: rusttype::Glyph) -> Vec<Vec2d> {
    let g = g.scaled(rusttype::Scale::uniform(100e0));
    let g = g.positioned(rusttype::point(0e0, 0e0));

    let mut collector = PointCollector::new(0.01e0);
    g.build_outline(&mut collector);

    collector.collected_points
}

fn cosine_interpolation(
    data: &Vec<f64>,
    time_range: &std::ops::RangeInclusive<f64>,
) -> splines::Spline<f64, f64> {
    let N = data.len() as f64;

    let delta_time = (time_range.end() - time_range.start()) / (N - 1e0);

    let mut time = *time_range.start();
    let mut t = || {
        let pre_incremented_time = time;
        time += delta_time;
        pre_incremented_time
    };

    let keys = data
        .iter()
        .map(|x| splines::Key::new(t(), *x, splines::Interpolation::Cosine))
        .collect();
    splines::Spline::from_vec(keys)
}

fn interpolation(
    data: &Vec<Vec2d>,
    time_range: &std::ops::RangeInclusive<f64>,
) -> splines::Spline<f64, Vec2d> {
    let N = data.len() as f64;

    let delta_time = (time_range.end() - time_range.start()) / (N - 1e0);

    let mut time = *time_range.start();
    let mut t = || {
        let pre_incremented_time = time;
        time += delta_time;
        pre_incremented_time
    };

    let keys = data
        .iter()
        .map(|x| splines::Key::new(t(), *x, splines::Interpolation::Cosine))
        .collect();
    splines::Spline::from_vec(keys)
}

fn text_to_equation(text: &str) -> anyhow::Result<String> {
    let mut result = String::new();

    const font_bytes: &[u8] = include_bytes!("../font.ttf");
    let font = rusttype::Font::try_from_bytes(font_bytes).ok_or(anyhow!("Couldn't load font!"))?;
    let mut points_2 = text_to_glyphs(&font, text)
        .into_iter()
        .map(|glyph| glyph_to_points(glyph))
        .collect::<Vec<_>>();

    let mut start = 0e0;
    let ts = 0e0..=100e0;
    for (n, mut points) in points_2.into_iter().enumerate() {
        ScaleVec2d::scale(&mut points, Some(((start, start + 10e0), (10e0, 0e0))));
        let xs = points.iter().map(|vec| vec.x).collect();
        let ys = points.iter().map(|vec| vec.y).collect();
        let approx_x = cosine_interpolation(&xs, &ts);
        let approx_y = cosine_interpolation(&ys, &ts);
        let fx = FourierSeries::new(
            FourierCoefficients::ApproxFn(Box::new(move |t| approx_x.sample(t).unwrap()), 500, &ts),
            10e0,
        );
        let fy = FourierSeries::new(
            FourierCoefficients::ApproxFn(Box::new(move |t| approx_y.sample(t).unwrap()), 500, &ts),
            10e0,
        );
        result.write_fmt(format_args!("x{}(t) = {}\n\n", n, fx.equation()?))?;
        result.write_fmt(format_args!("y{}(t) = {}\n\n", n, fy.equation()?))?;
        start += 12e0;
        result.write_str("\n\n\n")?;
    }

    Ok(result)
}

fn main() -> anyhow::Result<()> {
    println!("{}", text_to_equation("Balen")?);

    Ok(())
}
