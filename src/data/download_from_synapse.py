import os
import click
import synapseutils
import synapseclient

def synapse_login(username, password):
    """
    Creates an authenticated Synapse client and returns that client.
    """
    syn_client = synapseclient.Synapse()
    syn_client.login(username, password)
    return syn_client


def download_files(destination, syn_ids, syn_client):
    """
    Downloads the files corresponding to the Synapse IDs in `syn_ids`, to the
    given destination path. Requires an authenticated Synapse client.
    """
    os.makedirs(destination, exist_ok=True)
    for syn_id in syn_ids:
        click.echo("Downloading %s..." % syn_id)
        synapseutils.syncFromSynapse(syn_client, syn_id, path=destination)


@click.command()
@click.option(
    "--username", "-u", nargs=1, required=True, help="Synapse username"
)
@click.option(
    "--password", "-p", nargs=1, required=True, help="Synapse password"
)
@click.option(
    "--destination", "-d", nargs=1, default=".", type=click.Path(),
    help="Path to save downloads"
)
@click.argument("syn_ids", nargs=-1, required=True)
def main(username, password, destination, syn_ids):
    syn_client = synapse_login(username, password)
    download_files(destination, syn_ids, syn_client)


if __name__ == "__main__":
    main()
