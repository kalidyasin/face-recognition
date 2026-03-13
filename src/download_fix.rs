fn check_download(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Downloading {} (this may take a while)...", path);
        // Use a more robust client configuration
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600)) // 10 minute timeout for 250MB
            .build()?;
        
        let mut response = client.get(url).send()?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to download {}: HTTP {}", path, response.status()));
        }

        let mut file = std::fs::File::create(path)?;
        std::io::copy(&mut response, &mut file)?;
        
        let metadata = std::fs::metadata(path)?;
        println!("  - Finished. Size: {} MB", metadata.len() / 1024 / 1024);
    }
    Ok(())
}
