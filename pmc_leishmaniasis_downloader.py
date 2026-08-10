"""
Script de recherche et telechargement d'images cliniques de leishmaniose
cutanee depuis les case reports en libre acces sur PubMed Central (PMC).

Utilisation (invite de commande / terminal VS Code) :
    pip install requests
    python pmc_leishmaniasis_downloader.py

Ce que fait le script :
1. Cherche les articles PMC en texte integral libre mentionnant
   "cutaneous leishmaniasis" + "case report".
2. Pour chaque article trouve, recupere la liste de ses figures via l'API
   OA (Open Access) de PMC.
3. Telecharge les images de figures dans un dossier local, avec un fichier
   CSV recapitulatif (article, legende de la figure, fichier local).

Notes :
- Respecte les limites de l'API NCBI (max 3 requetes/seconde sans cle API).
- Toutes les images ne seront pas forcement des photos cliniques utiles
  (certaines figures sont des graphiques, histologie, cartes, etc.) :
  un tri manuel rapide du dossier de sortie est recommande apres coup.
"""

import os
import csv
import time
import re
import requests
import xml.etree.ElementTree as ET

# ----------------------- Configuration -----------------------

SEARCH_TERM = (
    '"cutaneous leishmaniasis"[Title/Abstract] '
    'AND "case report"[Title/Abstract] '
    'AND free fulltext[filter]'
)
MAX_ARTICLES = 60          # nombre d'articles PMC a explorer
OUTPUT_DIR = "leishmaniasis_pmc_images"
CSV_PATH = os.path.join(OUTPUT_DIR, "images_metadata.csv")
DELAY_SECONDS = 0.4        # pause entre requetes (reste sous 3 req/s)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_ARTICLE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"

HEADERS = {"User-Agent": "Mozilla/5.0 (research script; contact: user)"}


# ----------------------- Etape 1 : recherche -----------------------

def search_pmc_ids(term, max_results):
    """Renvoie une liste d'IDs PMC (sans le prefixe PMC) correspondant a la recherche."""
    params = {
        "db": "pmc",
        "term": term,
        "retmax": max_results,
        "retmode": "json",
    }
    r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


# ----------------------- Etape 2 : figures d'un article -----------------------

def get_article_figures(pmcid):
    """
    Recupere la page HTML de l'article PMC et extrait les images de figures.
    Utilise le rendu HTML public de PMC (plus fiable que l'API OA pour les figures).
    Renvoie une liste de dicts {url, caption}.
    """
    url = PMC_ARTICLE_URL.format(pmcid=pmcid)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return []

    html = r.text
    figures = []

    # Cherche les blocs <img ... src="...bin/....jpg"> typiques des figures PMC
    img_matches = re.findall(r'<img[^>]+src="([^"]+/bin/[^"]+\.(?:jpg|jpeg|png|gif))"', html, re.IGNORECASE)
    for img_url in img_matches:
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            img_url = "https://www.ncbi.nlm.nih.gov" + img_url
        figures.append({"url": img_url, "caption": ""})

    return figures


# ----------------------- Etape 3 : telechargement -----------------------

def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def download_image(url, dest_path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except requests.RequestException as e:
        print(f"  [erreur telechargement] {url} -> {e}")
        return False


# ----------------------- Programme principal -----------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Recherche PMC : {SEARCH_TERM}")
    pmc_ids = search_pmc_ids(SEARCH_TERM, MAX_ARTICLES)
    print(f"{len(pmc_ids)} articles trouves.\n")

    rows = []
    total_images = 0

    for i, pmcid in enumerate(pmc_ids, start=1):
        print(f"[{i}/{len(pmc_ids)}] Article PMC{pmcid}...")
        figures = get_article_figures(pmcid)
        time.sleep(DELAY_SECONDS)

        if not figures:
            print("  Aucune figure trouvee.")
            continue

        for j, fig in enumerate(figures, start=1):
            ext = os.path.splitext(fig["url"])[1] or ".jpg"
            filename = sanitize_filename(f"PMC{pmcid}_fig{j}{ext}")
            dest_path = os.path.join(OUTPUT_DIR, filename)

            ok = download_image(fig["url"], dest_path)
            if ok:
                total_images += 1
                rows.append({
                    "pmcid": f"PMC{pmcid}",
                    "article_url": PMC_ARTICLE_URL.format(pmcid=pmcid),
                    "figure_url": fig["url"],
                    "local_file": filename,
                })
                print(f"  OK -> {filename}")
            time.sleep(DELAY_SECONDS)

    # Ecrit le CSV recapitulatif
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pmcid", "article_url", "figure_url", "local_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTermine. {total_images} images telechargees dans '{OUTPUT_DIR}/'.")
    print(f"Recapitulatif : {CSV_PATH}")
    print("\nATTENTION : certaines figures ne sont pas des photos cliniques")
    print("(graphiques, cartes, histologie). Fais un tri visuel rapide du dossier.")


if __name__ == "__main__":
    main()