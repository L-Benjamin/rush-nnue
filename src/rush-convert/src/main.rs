#![allow(unused)]

use std::fs::File;
use std::io::Write;

use anyhow::{Error, Result};
use clap::{App, Arg};
use serde::Deserialize;

/// Shorthand type for a dynamic matrix of f32.
type Mat = Vec<Vec<f32>>;

/// Struct read from a json file.
#[derive(Deserialize)]
struct Net {
    w0: Mat,
    b0: Vec<f32>,
    w1: Mat,
    b1: Vec<f32>,
    w2: Mat,
    b2: Vec<f32>,
    w3: Mat,
    b3: Vec<f32>,
}

/// Transpose a matrix in place.
fn transpose(mat: &mut Mat) {
    let dim0 = mat.len();
    let dim1 = mat[0].len();

    let mut res = vec![vec![0.0; dim0]; dim1];

    for i in 0..dim0 {
        for j in 0..dim1 {
            res[j][i] = mat[i][j];
        }
    }

    *mat = res;
}

/// Write all the elements of a vector in big endian.
fn write_vec<W: Write>(writer: &mut W, vec: &Vec<f32>) -> Result<()> {
    for element in vec {
        writer.write(&element.to_be_bytes())
            .map_err(|_| Error::msg("Unable to write to ouput file"))?;
    }

    Ok(())
}

/// Write all the elements of a matrix in big endian.
fn write_mat<W: Write>(writer: &mut W, mat: &Mat) -> Result<()> {
    for row in mat {
        write_vec(writer, row)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    // Parse the arguments.
    let args = App::new("Rush NNUE converter")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Benjamin Lefebvre")
        .about("Converts a NNUE from a JSON file to a special format usable by the rush chess engine")
        .arg(Arg::with_name("json")
            .index(1)
            .value_name("JSON")
            .help("The path to the JSON file to convert.")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("out")
            .index(2)
            .value_name("OUT")
            .help("The path of the output file.")
            .takes_value(true)
            .required(true))
        .get_matches();
    
    // Open the json file.
    let json = File::open(args.value_of("json").unwrap())
        .map_err(|_| Error::msg("Cannot open JSON file."))?;

    // Create the output file.
    let mut out = File::create(args.value_of("out").unwrap())
        .map_err(|_| Error::msg("Cannot create output file."))?;

    // Parse the net.
    let mut net: Net = serde_json::from_reader(&json)
        .map_err(|_| Error::msg("Cannot parse JSON file."))?;

    // Transpose the matrices, to align the floats. This makes inference more efficient.
    transpose(&mut net.w0);
    transpose(&mut net.w1);
    transpose(&mut net.w2);
    transpose(&mut net.w3);

    // Write all the nets data.
    write_mat(&mut out, &net.w0);
    write_vec(&mut out, &net.b0);
    write_mat(&mut out, &net.w1);
    write_vec(&mut out, &net.b1);
    write_mat(&mut out, &net.w2);
    write_vec(&mut out, &net.b2);
    write_mat(&mut out, &net.w3);
    write_vec(&mut out, &net.b3);

    Ok(())
}
